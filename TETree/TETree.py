import random
import time

from CSRGraph import edgelist_to_CSR_gpu2, edgelist_and_truss_to_csr_gpu, read_edge_txt_gpu2, read_edge_and_truss_txt_gpu
from utils import get_all_nbr, device, sp_edge_unique, calculate_time, cpu, sp_edge_unique2, sp_edge_unique_mask, \
    sp_edge_unique3
# from preprocessing_utils import calucate_triangle_save, calucate_triangle_nosave, calucate_triangle_cpu
# from truss_no_save3 import truss_decomposition
from torch_scatter import segment_csr
import torch


batch = 8000000  # 三角形计算batch
off = 1000000000   # off设置1000万，大数据集可能不够用
src_batch = 50000000 # 设置src溢出batch的标准，占据381MB


def intersection(values, boundaries): #mask有序，value无序
    mask = values<=boundaries[-1]
    mask1 = torch.nonzero(mask).squeeze(1)
    values = values[mask1]
    result = torch.bucketize(values, boundaries)
    mask[mask1] = boundaries[result]==values
    return mask

def get_all_nbr_in(starts: torch.Tensor, nexts: torch.Tensor):
    """
    :param starts:[1,1,1,5,5]
    :param nexts: [4,4,4,8,8]
    :return:
    """
    sizes = nexts - starts
    nbr_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(sizes, dim=0, dtype=torch.int32)))  # [0,3,6,9,12,15]
    # 从索引到点
    nbr = torch.arange(int(nbr_ptr[-1]), device=device, dtype=torch.int32) - torch.repeat_interleave(nbr_ptr[:-1] - starts, sizes)
    return nbr, nbr_ptr, sizes



@calculate_time
def equi_tree_construction(row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    """
    为了保证truss值低的pi值更小
    """
    time1 = time.time()
    pi = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    
    truss_sort, indices = torch.sort(truss_result, stable=True)
    edge_sort = edge_id[indices]  # indices为int64，作为索引来讲，可以使用
    _, edge_indices = torch.sort(edge_sort, stable = True)  # 按照truss排序后的边再重新排回原位
    edge_indices = edge_indices.to(torch.int32) # edge_indices就指代边在truss排序中的位置
    border_edge_mask = torch.zeros(columns.size(0), device=device, dtype=torch.bool)

    del truss_sort, indices, edge_sort, _

    sizes = (row_ptr[1:] - row_ptr[:-1])
    # columns中每个点的度数，其中columns即为每轮batch的限制，columns为uvw中的中间点v
    max_nbr_count = segment_csr(sizes[columns].to(torch.int64), row_ptr.to(torch.int64))  # 长度与row_ptr相同, 类型已经为int64了

    max_nbr_count = max_nbr_count.cumsum(0)  # 求和的值太大了，要检查是否溢出了
    # 获取插入的索引
    group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch), step=batch, dtype=torch.int64,
                                                           device=device), side='right')

    for head, tail in zip(group[0:-1], group[1:]):
        if head == tail:
            continue
        u_vertices = torch.arange(head, tail, device=device, dtype=torch.int32)

        # 获取u的邻居v
        u_nbr_indices = torch.arange(int(row_ptr[head]), int(row_ptr[tail]), device=device, dtype=torch.int32)
        v_vertices = columns[row_ptr[head]: row_ptr[tail]]

        # 计算u.v即直连的u.w
        u_repeat = torch.repeat_interleave(u_vertices, row_ptr[head + 1: tail + 1] - row_ptr[head: tail])
        u_w = u_repeat.type(torch.int64) * off + v_vertices.type(torch.int64)

        # 获取v的邻居w
        v_nbr_indices, v_nbr_ptr, v_nbr_sizes = get_all_nbr_in(row_ptr[v_vertices], row_ptr[v_vertices + 1])
        u_v_nbr = columns[v_nbr_indices]

        # 计算u.w，经过v相连的u.w
        u_v_repeat = torch.repeat_interleave(u_repeat, v_nbr_sizes)
        u_v_w = u_v_repeat.type(torch.int64) * off + u_v_nbr.type(torch.int64)

        uvw_mask = intersection(u_v_w, u_w)

        batch_support = segment_csr(uvw_mask.to(torch.int64), v_nbr_ptr.to(torch.int64)).to(torch.int32)  # 算的是uv的支持度（一个uv下边有几个共同的w点）

        s_e = torch.repeat_interleave(u_nbr_indices, batch_support)
        l_e = v_nbr_indices[uvw_mask]
        r_e = torch.bucketize(u_v_w[uvw_mask], u_w) + row_ptr[head]
        
        del u_vertices, u_nbr_indices, v_vertices, u_repeat, u_w, v_nbr_indices,  v_nbr_ptr, v_nbr_sizes, u_v_nbr, u_v_repeat, u_v_w, uvw_mask, batch_support
        mask1 = truss_result[s_e] == truss_result[l_e]
        mask2 = truss_result[s_e] == truss_result[r_e]
        mask1 &= mask2
        mask = ~mask1
        s_e2 = s_e[mask]  # 按从小到大的顺序排列
        l_e2 = l_e[mask]
        r_e2 = r_e[mask]
        s_e = s_e[mask1]
        l_e = l_e[mask1]
        r_e = r_e[mask1]

        border_edge_mask[s_e2] = 1  # 记这些边为边界边
        temp = torch.stack((s_e2, l_e2, r_e2))
        _, ind = torch.sort(truss_result[temp], dim=0)
        temp = temp[ind, torch.arange(0, temp.size(1), device=device, dtype=torch.int32)]
        s_e2 = temp[0]
        l_e2 = temp[1]
        r_e2 = temp[2]
        # r_e一定是truss值更大的边
        mask = truss_result[l_e2] == truss_result[s_e2]

        merge_src = edge_indices[torch.cat((s_e, s_e, s_e2[mask]))]   # src小
        merge_des = edge_indices[torch.cat((l_e, r_e, l_e2[mask]))]   # des大

        merge_src, ind = torch.sort(merge_src, descending=True)
        merge_des = merge_des[ind]
        merge_des, ind = torch.sort(merge_des, stable=True, descending=True)
        merge_src = merge_src[ind]


        link(merge_des.flip(0), merge_src.flip(0), pi)

        del s_e, l_e, r_e, s_e2, r_e2, l_e2, mask, mask1, mask2, temp, _, ind, merge_src, merge_des


    del max_nbr_count, group
    compress(edge_id, pi)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


    edge_to_node = torch.full((columns.size(0),), -1, device=device, dtype=torch.long)
    '''
    筛选掉孤立边
    '''
    mask = truss_result != 2
    valid_id = edge_id[mask]  # 【2，3，4，5，6，8】
    pi_old = pi[edge_indices[valid_id]]

    '''
    超节点编号本身以edgeid编号，但edgeid的数量远大于spnode的数量，
    因此进行重新编号
    '''
    # 使用return_inverse即可返回pi中元素在unique中的索引，unique本身排过序了，索引即从0开始重新编号
    # 这时候就已经从小到大开始编号了，以edgeindices为序列的边
    unique_spnode, pi = torch.unique(pi_old, return_inverse=True)

    max_node_id = unique_spnode.size(0) - 1
    edge_to_node[valid_id] = pi

    # print("spnode num", unique_spnode.size(0))
    del unique_spnode, mask, pi_old


    src_edge, des_edge = calucate_super_edges(columns, rows, row_ptr, edge_to_node, edge_id, truss_result, border_edge_mask, edge_indices)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())
    # print("spedge num", src_edge.size(0))
    #
    # print(max_node_id)
    # print(torch.max(edge_to_node))
    # print(torch.max(des_edge))

    time2 = time.time()
    # print("truss时间:",time2-time1)
    #
    # print("开始进行equitree构建")
    node_to_tao = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    tree_pi = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    node_id = tree_pi.clone()
    node_to_tao[pi] = truss_result[valid_id]


    src_node = des_edge   # 大k
    des_node = src_edge   # 小k
    i=0
    while True:
        i+=1
        # print("==tree",i,"==")

        bound = torch.cat((torch.tensor([1], device=device, dtype=torch.int32) ,src_node[1:] - src_node[:-1]))
        bound = bound.to(torch.bool)
        selected_indices = torch.where(~bound)[0]
        if selected_indices.size(0)==0:
            break
        src_node[selected_indices] = des_node[selected_indices-1]

        src_node, des_node = sp_edge_unique2(src_node, des_node)

        mask = node_to_tao[src_node] == node_to_tao[des_node]
        merge_src = src_node[mask]
        merge_des = des_node[mask]
        # 在这里的link中，merge_src和merge_des从小到大排序，且src>des
        # 这样的数据结构对link更友好
        link(merge_src, merge_des, tree_pi)
        compress(torch.cat((merge_src, merge_des)), tree_pi)

        src_node = tree_pi[src_node[~mask]]  # 提前过滤掉(横向)的边
        des_node = tree_pi[des_node[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        src_node, des_node = sp_edge_unique3(src_node, des_node)

    compress(node_id, tree_pi)
    unique_spnode, tree_pi = torch.unique(tree_pi, return_inverse=True)
    edge_to_node = tree_pi[edge_to_node]
    max_node_id = unique_spnode.size(0)-1
    src_node = tree_pi[src_node]
    des_node = tree_pi[des_node]

    time3 = time.time()

    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_node.size(0))
    print("equitree构造时间：", time3-time1, "!!!!")
    print("equitruss构造时间：", time2-time1, "!!!!")

    # 返回超节点数据
    return valid_id, edge_to_node, src_edge, des_edge, max_node_id



def calucate_super_edges(columns: torch.Tensor, rows: torch.Tensor, row_ptr: torch.Tensor, pi: torch.Tensor,
                          edge_id: torch.Tensor, truss_result: torch.Tensor, border_edge_mask: torch.Tensor,
                          edge_indices: torch.Tensor ):

    src_edge = torch.tensor([], device=device, dtype=torch.int32)
    des_edge = torch.tensor([], device=device, dtype=torch.int32)

    sizes = (row_ptr[1:] - row_ptr[:-1])
    # columns中每个点的度数，其中columns即为每轮batch的限制，columns为uvw中的中间点v
    max_nbr_count = segment_csr(sizes[columns].to(torch.int64), row_ptr.to(torch.int64))  # 长度与row_ptr相同, 类型已经为int64了

    max_nbr_count = max_nbr_count.cumsum(0)  # 求和的值太大了，要检查是否溢出了
    # 获取插入的索引
    group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch), step=batch, dtype=torch.int64,
                                                           device=device), side='right')
    i=0
    for head, tail in zip(group[0:-1], group[1:]):
        if head == tail:
            continue
        u_vertices = torch.arange(head, tail, device=device, dtype=torch.int32)

        # 获取u的邻居v
        u_nbr_indices = torch.arange(int(row_ptr[head]), int(row_ptr[tail]), device=device, dtype=torch.int32)
        v_vertices = columns[row_ptr[head]: row_ptr[tail]]

        # 计算u.v即直连的u.w
        u_repeat = torch.repeat_interleave(u_vertices, row_ptr[head + 1: tail + 1] - row_ptr[head: tail])
        u_w = u_repeat.type(torch.int64) * off + v_vertices.type(torch.int64)

        # 获取v的邻居w
        v_nbr_indices, v_nbr_ptr, v_nbr_sizes = get_all_nbr_in(row_ptr[v_vertices], row_ptr[v_vertices + 1])
        u_v_nbr = columns[v_nbr_indices]

        # 计算u.w，经过v相连的u.w
        u_v_repeat = torch.repeat_interleave(u_repeat, v_nbr_sizes)
        u_v_w = u_v_repeat.type(torch.int64) * off + u_v_nbr.type(torch.int64)

        uvw_mask = intersection(u_v_w, u_w)

        batch_support = segment_csr(uvw_mask.to(torch.int64), v_nbr_ptr.to(torch.int64)).to(torch.int32)  # 算的是uv的支持度（一个uv下边有几个共同的w点）

        s_e = torch.repeat_interleave(u_nbr_indices, batch_support)
        l_e = v_nbr_indices[uvw_mask]
        r_e = torch.bucketize(u_v_w[uvw_mask], u_w) + row_ptr[head]

        mask1 = truss_result[s_e] == truss_result[l_e]
        mask2 = truss_result[s_e] == truss_result[r_e]
        mask1 &= mask2
        mask1 = ~mask1
        s_e2 = s_e[mask1]  # 按从小到大的顺序排列
        l_e2 = l_e[mask1]
        r_e2 = r_e[mask1]
        temp = torch.stack((s_e2, l_e2, r_e2))
        _, ind = torch.sort(truss_result[temp], dim=0)
        temp = temp[ind, torch.arange(0, temp.size(1), device=device, dtype=torch.int32)]
        s_e2 = temp[0]
        l_e2 = temp[1]
        r_e2 = temp[2]
        # r_e一定是truss值更大的边
        mask = truss_result[l_e2] == truss_result[s_e2]
        src_edge = torch.cat((src_edge, pi[torch.cat((s_e2[~mask], s_e2))]))
        des_edge = torch.cat((des_edge, pi[torch.cat((l_e2[~mask], r_e2))]))
        if src_edge.size(0)>src_batch or tail == group[-1]:
            i+=1
            # print("===去重:",i,"===")
            des_edge, src_edge = sp_edge_unique3(des_edge, src_edge)
        del s_e, l_e, r_e, s_e2, r_e2, l_e2, mask, mask1, mask2, temp, _, ind
    
    return src_edge, des_edge


# afforest cc link
def link(e: torch.Tensor, e1: torch.Tensor, pi: torch.Tensor):
    p1 = pi[e]
    p2 = pi[e1]
    # mask = p1 != p2

    # 计算数量
    while p1.size(0) > 0:
        mask = p2 >= p1
        h = torch.where(mask, p2, p1)  # 大标签
        l = p1 + p2 - h  # 小标签

        # 判断是否已收敛到祖先
        mask1 = pi[h] == h

        # 重点在这，祖先可能被变化多次，只会以最后一次为主
        pi[h[mask1]] = l[mask1]  # 是则将祖先指向较小的那个

        # 被变化多次的祖先中，只筛选已经相等并收敛的
        mask2 = (pi[h] == l) & mask1

        h = h[~mask2]
        l = l[~mask2]

        p2 = pi[pi[h]]
        p1 = pi[l]
        # mask = p1 != p2


# afforest cc compress算法
def compress(e: torch.Tensor, pi: torch.Tensor):
    # 除非已经收敛，否则每条边都不断往上更新到祖先的标签值
    while (pi[pi[e]] == pi[e]).sum().item() != e.shape[0]:
        pi[e] = pi[pi[e]]




def run_with_truss(filename: str, name: str = ""):
    torch.cuda.empty_cache()
    print("=======!!!{}=======".format(name))
    edge_starts, edge_ends, truss = read_edge_and_truss_txt_gpu(filename, 0)
    row_ptr, columns, rows, truss_result = edgelist_and_truss_to_csr_gpu(edge_starts, edge_ends, truss, direct=True)
    del edge_starts, edge_ends, truss

    torch.cuda.empty_cache()
    row_ptr = row_ptr.to(device)
    columns = columns.to(device)
    rows = rows.to(device)
    truss_result = truss_result.to(device)

    print("开始equitree构建")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    equi_tree_construction(row_ptr, columns, rows, truss_result)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run EquiTree construction with Graph file')
    parser.add_argument('--filename', '-f',
                        default=r"/home/featurize/work/TETree/facebook_truss_result.txt",
                        help='Path to the truss result file')
    parser.add_argument('--name', '-n', default="facebook_truss",
                        help='Name for the run (optional)')

    args = parser.parse_args()
    run_with_truss(args.filename, name=args.name)

    
