import time

from CSRGraph import edgelist_to_CSR_gpu2, edgelist_and_truss_to_csr_gpu, read_edge_txt_gpu2, read_edge_and_truss_txt_gpu
from utils import get_all_nbr, device, sp_edge_unique, calculate_time, cpu, sp_edge_unique2, sp_edge_unique_mask, \
    sp_edge_unique3
from torch_scatter import segment_csr
# from line_profiler import LineProfiler
# from trusstensor import segment_triangle_isin, segment_direct_isin
# import equitree_one_step1
# from truss_no_save3 import truss_decomposition, calculate_triangles

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = 20000000
off = 1000000000  # offset偏移值，用来联合isin

def intersection(values, boundaries):  # value和mask都有序
    mask = values <= boundaries[-1]  # 这个是顺序的，应该可以再次加速的
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result] == values
    return mask


def get_all_nbr_in(starts: torch.Tensor, nexts: torch.Tensor):
    """
    :param starts:[1,1,1,5,5]
    :param nexts: [4,4,4,8,8]
    :return:
    """
    sizes = nexts - starts
    nbr_ptr = torch.cat((torch.tensor([0], device=device), torch.cumsum(sizes, dim=0, dtype=torch.int32)))  # [0,3,6,9,12,15]
    # print(nbr_ptr.size(0))
    # 从索引到点
    nbr = torch.arange(int(nbr_ptr[-1]), device=device, dtype=torch.int32) - torch.repeat_interleave(nbr_ptr[:-1] - starts, sizes)
    return nbr, nbr_ptr, sizes

# 加入超节点与边之间的映射
@calculate_time
# @profile
def equi_tree_construction(row_ptr: torch.Tensor, columns: torch.Tensor, truss_result: torch.Tensor):
    t0 = time.time()
    source_vertices = torch.repeat_interleave(torch.arange(0, row_ptr.size(0)-1, device=device, dtype=torch.int32)
                                              , row_ptr[1:]-row_ptr[:-1])
    pi = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    real_pi = pi.clone()

    node_id = pi.clone()
    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    src_edge = torch.tensor([], device=device, dtype=torch.int32)
    des_edge = torch.tensor([], device=device, dtype=torch.int32)

    truss_sort, indices = torch.sort(truss_result, stable=True)
    k_list, counts = torch.unique_consecutive(truss_sort, return_counts=True)

    edge_sort = edge_id[indices]
    del indices
    
    '''
    indices指示按truss排过序的边在pi中的位置
    '''
    _, indices = torch.sort(edge_sort)
    del _

    edge_ptr = torch.cat([torch.tensor([0], device=device, dtype=torch.int32),
                          torch.cumsum(counts, dim=0, dtype=torch.int32)])

    # triangle_count = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
    # segment_direct_isin(source_vertices, columns, row_ptr, triangle_count)

    # pi = edge_sort.clone()
    # sizes = (row_ptr[1:] - row_ptr[:-1])
    # nbr_counts = segment_csr(sizes[columns].to(torch.int64), row_ptr.to(torch.int64))  # 长度与row_ptr相同, 类型已经为int64了
    # nbr_counts = sizes[columns] # 每条边对应的点的领居数量的点邻居数量

    # nbr_counts = nbr_counts.cumsum(0)  # 求和的值太大了，要检查是否溢出了



    i = int(counts.size(0)) - 1
    k_list = k_list.flip(dims=[0])
    del counts

    sp_node_time = 0
    sp_edge_time = 0
    equitree_time = 0
    iteration_count = 0

    for k in k_list:
        t1 = time.time()
        phi_k = edge_sort[edge_ptr[i]: edge_ptr[i + 1]]
        i -= 1

        u = source_vertices[phi_k]
        v = columns[phi_k]

        p = torch.unique(torch.cat([u, v]))  # p即为所有点
        mask_v = torch.zeros(row_ptr.shape[0], dtype=torch.bool, device=device)
        mask_v[p] = True
        mask = mask_v[columns]  # 查找点的入边,-1一定为false，不会被选择
        p_c = get_all_nbr(row_ptr[p], row_ptr[p + 1])  # 出边索引
        mask[p_c] = True

        del u, v, p, p_c, mask_v
        # print("mask", torch.sum(mask.to(torch.int32)))

        sub_columns = columns[mask]  # 这里可以试试，用torch.nonzero(mask).squeeze(1)还是不用快
        sub_row_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.int32),
                             torch.cumsum(segment_csr(mask.to(torch.int64), row_ptr.to(torch.int64)).to(torch.int32),
                                          dim=0, dtype=torch.int32)))
        sub_edges = torch.arange(columns.shape[0], dtype=torch.int32, device=device)[mask]  # 我在子图中计算三角形


        triangle_edge_mask = torch.ones(columns.shape[0], dtype=torch.bool, device=device)
        triangle_edge_mask[phi_k] = False

        sizes = (sub_row_ptr[1:] - sub_row_ptr[:-1])
        # columns中每个点的度数，其中columns即为每轮batch的限制，columns为uvw中的中间点v
        max_nbr_count = segment_csr(sizes[sub_columns].to(torch.int64), sub_row_ptr.to(torch.int64))  # 长度与row_ptr相同
        max_nbr_count = max_nbr_count.cumsum(0)  # 求和的值太大了，要检查是否溢出了
        group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch), step=batch, dtype=torch.int64,
                                                           device=device), side='right')

        k_src_edge = torch.tensor([], device=device, dtype=torch.int32)
        k_des_edge = torch.tensor([], device=device, dtype=torch.int32)

        t2 = time.time()

        sp_node_time+= (t2-t1)/2
        sp_edge_time+= (t2-t1)/2
        # 每一波都顶着batch的极限去算
        for head, tail in zip(group[0:-1], group[1:]):
            if head == tail:
                # 说明这个点的邻居太多，很可能难以算完，但是没办法，只能在一次batch中计算
                # 所以continue直到tail后移一位，此时head还是不变，不会影响结果
                continue
            iteration_count+=1
            t3 = time.time()
            
            u_vertices = torch.arange(head, tail, device=device, dtype=torch.int32)
            # 获取u的邻居v
            u_nbr_indices = torch.arange(sub_row_ptr[head], sub_row_ptr[tail], device=device, dtype=torch.int32)
            v_vertices = sub_columns[sub_row_ptr[head]: sub_row_ptr[tail]]
            # 计算u.v即直连的u.w
            u_repeat = torch.repeat_interleave(u_vertices, sub_row_ptr[head + 1: tail + 1] - sub_row_ptr[head: tail])
            u_w = u_repeat.type(torch.int64) * off + v_vertices.type(torch.int64)


            # 获取v的邻居w
            v_nbr_indices, v_nbr_ptr, v_nbr_sizes = get_all_nbr_in(sub_row_ptr[v_vertices], sub_row_ptr[v_vertices + 1])

            u_v_nbr = sub_columns[v_nbr_indices]
            # 计算u.w，经过v相连的u.w
            u_v_repeat = torch.repeat_interleave(u_repeat, v_nbr_sizes)
            u_v_w = u_v_repeat.type(torch.int64) * off + u_v_nbr.type(torch.int64)

            uvw_mask = intersection(u_v_w, u_w)
            batch_support = segment_csr(uvw_mask.to(torch.int64), v_nbr_ptr.to(torch.int64)).to(torch.int32)  # 算的是uv的支持度（一个uv下边有几个共同的w点）

            # 总长度即为uvw_mask中true的数量
            s_e =  sub_edges[torch.repeat_interleave(u_nbr_indices, batch_support)]
            # 列出vw
            l_e = sub_edges[v_nbr_indices[uvw_mask]]
            # 列出uw
            r_e =sub_edges[torch.bucketize(u_v_w[uvw_mask], u_w) + sub_row_ptr[head]]
            del u_vertices, u_nbr_indices, v_vertices, u_repeat, u_w, v_nbr_indices,  v_nbr_ptr, v_nbr_sizes, u_v_nbr, u_v_repeat, u_v_w, uvw_mask, batch_support
            
            # 非phix三角形三边
            mask = triangle_edge_mask[s_e] & triangle_edge_mask[l_e] & triangle_edge_mask[r_e]
            # 变为受到删除边影响的三边（只有与删除边有联系，才会被计数）
            mask = ~mask
            s_e = s_e[mask]
            l_e = l_e[mask]
            r_e = r_e[mask]
            del mask
            
            
            temp = torch.stack((s_e, l_e, r_e))
            _, ind = torch.sort(truss_result[temp], dim=0)
            temp = temp[ind, torch.arange(0, temp.size(1), device=device)]
            s_e = temp[0]
            l_e = temp[1]
            r_e = temp[2]
            del temp, _, ind

            mask = truss_result[s_e] == k
            s_e = s_e[mask]
            l_e = l_e[mask]
            r_e = r_e[mask]
            t4 = time.time()

            # afforest
            # k-triangle connectivity
            k1 = truss_result[l_e]
            k2 = truss_result[r_e]
            mask1 = k1 == k
            # link(indices[s_e[mask1]], indices[l_e[mask1]], pi)
            link(indices[s_e[mask1]], indices[l_e[mask1]], pi)
            mask2 = k2 == k
            link(indices[s_e[mask2]], indices[r_e[mask2]], pi)
            compress(indices[phi_k], pi)

            t5 = time.time()

            mask1 = (k1 > k)
            mask2 = (k2 > k)
            k_src_edge = torch.cat((k_src_edge, pi[indices[s_e[mask2]]], pi[indices[s_e[mask1]]]))
            k_des_edge = torch.cat((k_des_edge, pi[indices[r_e[mask2]]], pi[indices[l_e[mask1]]]))

            if k_src_edge.size(0) > 0:
                k_src_edge, k_des_edge = sp_edge_unique2(k_src_edge, k_des_edge)

            t6 = time.time()
            sp_node_time += t5-t4+(t4-t3)/2
            sp_edge_time += t6-t5+(t4-t3)/2
            del s_e, r_e, l_e, mask1, mask2

            torch.cuda.empty_cache()

        if k_src_edge.size(0) > 0:
            t7 = time.time()
            link(k_src_edge, k_des_edge, pi)  # 进行连通分量查找
            compress(edge_id, pi)
            src_edge = torch.cat([src_edge, pi[k_src_edge]])
            des_edge = torch.cat([des_edge, k_des_edge])
            src_edge, des_edge = sp_edge_unique2(src_edge, des_edge)
            t8 = time.time()
            equitree_time += (t8-t7)
            # 只更新该k值里合并的节点
            del k_src_edge, k_des_edge
        src_edge = pi[src_edge]
        real_pi[indices[phi_k]] = pi[indices[phi_k]]
        del max_nbr_count, group, phi_k, sizes, sub_columns, sub_row_ptr, sub_edges, triangle_edge_mask
        torch.cuda.empty_cache()


    print("循环次数：", iteration_count)
    print("spnode时间：", sp_node_time)
    print("spedge时间：", sp_edge_time)
    print("equitree转换时间：", equitree_time)
    '''
    构造一个新的并查集，为-1的边在超边构造中不会访问到，冗余
    '''
    edge_to_node = torch.full((columns.size(0),), -1, device=device, dtype=torch.int32)

    '''
    筛选掉孤立边
    '''
    mask = truss_result > 2
    # 争取的indices的位置被筛选出来
    pi_old = real_pi[indices[mask]]
    '''
    这些truss有效的边的pi值并不在原位，通过indices调整顺序之后，放到pi_old里
    此时此刻，valid_id与pi_old为一一对应
    '''
    valid_id = edge_id[mask]

    '''
    超节点编号本身以edgeid编号，但edgeid的数量远大于spnode的数量，
    因此进行重新编号
    '''
    # 使用return_inverse即可返回pi中元素在unique中的索引，unique本身排过序了，索引即从0开始重新编号
    unique_spnode, pi = torch.unique(pi_old, return_inverse=True)

    max_node_id = unique_spnode.size(0) - 1

    pi = pi.to(torch.int32)
    # clone过来
    edge_to_node[valid_id] = pi

    # 超图构建
    # 变为由小节点指向大节点的边
    '''
    由大k指向小k的点，树的逻辑在最后进行修改
    '''

    # 去重
    src_edge, des_edge = sp_edge_unique2(src_edge, des_edge)

    # 超节点的映射关系
    node_hash = torch.full((unique_spnode[-1] + 1,), -1, device=device, dtype=torch.int32)
    node_hash[pi_old] = pi
    src_edge = node_hash[src_edge]
    des_edge = node_hash[des_edge]

    tx = time.time()
    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_edge.size(0))
    print("equitree构造时间：", tx - t0, "!!!!")
    print("equitruss构造时间：", tx - t0 - equitree_time, "!!!!")


    return valid_id, edge_to_node, src_edge, des_edge, max_node_id



def link(e: torch.Tensor, e1: torch.Tensor, pi: torch.Tensor):
    p1 = pi[e]
    p2 = pi[e1]
    mask = p1 != p2

    # 计算数量
    while torch.sum(mask.to(torch.int32))!= 0:
        p1 = p1[mask]
        p2 = p2[mask]

        mask = p1 >= p2
        h = torch.where(mask, p1, p2)  # 大标签
        l = p1 + p2 - h  # 小标签

        # 判断是否已收敛到祖先
        mask1 = pi[h] == h

        # 重点在这，祖先可能被变化多次，只会以最后一次为主
        pi[h[mask1]] = l[mask1]  # 是则将祖先指向较小的那个

        # 被变化多次的祖先中，只筛选已经相等并收敛的
        mask2 = (pi[h] == l) & mask1

        h = h[~mask2]
        l = l[~mask2]

        p1 = pi[pi[h]]
        p2 = pi[l]
        mask = p1 != p2



# 对边或边集进行压缩操作
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

    equi_tree_construction(row_ptr, columns, truss_result)


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




