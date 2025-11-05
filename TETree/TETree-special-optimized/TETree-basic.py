import time

from CSRGraph import edgelist_to_CSR_gpu2, edgelist_and_truss_to_csr_gpu, read_edge_txt_gpu2, read_edge_and_truss_txt_gpu
from utils import get_all_nbr, device, sp_edge_unique_ascending 
from torch_scatter import segment_csr
from trusstensor import segment_triangle_isin, segment_direct_isin

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
在equitree9和equitruss_batch5.2的基础上进行，现算三角形。
'''

batch = 20000000  # 三角形计算batch

# 加入超节点与边之间的映射
def equi_tree_construction(row_ptr: torch.Tensor, columns: torch.Tensor, truss_result: torch.Tensor):
    time_start = time.time()
    source_vertices = torch.repeat_interleave(torch.arange(0, row_ptr.size(0)-1, device=device, dtype=torch.int32)
                                              , row_ptr[1:]-row_ptr[:-1])
    pi = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    real_pi = pi.clone()

    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    src_edge = torch.tensor([], device=device, dtype=torch.int32)
    des_edge = torch.tensor([], device=device, dtype=torch.int32)

    truss_sort, indices = torch.sort(truss_result, stable=True)
    k_list, counts = torch.unique_consecutive(truss_sort, return_counts=True)
    edge_sort = edge_id[indices]
    del truss_sort, indices
    _, indices = torch.sort(edge_sort)
    del _


    triangle_count = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
    segment_direct_isin(source_vertices, columns, row_ptr, triangle_count)



    edge_ptr = torch.cat([torch.tensor([0], device=device, dtype=torch.int32),
                          torch.cumsum(counts, dim=0, dtype=torch.int32)])

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

        del u, v, p, p_c
        # print("mask", torch.sum(mask.to(torch.int32)))

        sub_triangle_count = triangle_count[mask]
        sub_edge_id = edge_id[mask]
        max_nbr_count = sub_triangle_count.to(torch.int64).cumsum(0)  # 每条边的三角形
        group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch),
                                                           step=batch, dtype=torch.int64, device=device), side='right')
        max_nbr_count = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), max_nbr_count))

        k_src_edge = torch.tensor([], device=device, dtype=torch.int32)
        k_des_edge = torch.tensor([], device=device, dtype=torch.int32)

        t2 = time.time()

        sp_node_time+= (t2-t1)/2
        sp_edge_time+= (t2-t1)/2
        # 每一轮都按batch算三角形
        for head, tail in zip(group[0:-1], group[1:]):
            if head == tail:
                # 说明这个点的邻居太多，很可能难以算完，但是没办法，只能在一次batch中计算
                # 所以continue直到tail后移一位，此时head还是不变，不会影响结果
                continue
            iteration_count+=1
            t3 = time.time()
            sub_edges = sub_edge_id[head: tail]  # 被选中的边
            s_e = torch.repeat_interleave(sub_edges, sub_triangle_count[head: tail])
            u_nbr_ptr = max_nbr_count[head: tail]-max_nbr_count[head]

            l_e = torch.full((s_e.size(0),), -1, device=device, dtype=torch.int32)
            r_e = l_e.clone()
            # 通过subedges获取要处理的边，通过row_ptr获得两边端点，通过unbrptr获得三角形写入的位置
            segment_triangle_isin(source_vertices, columns, row_ptr, sub_edges, u_nbr_ptr.to(torch.int32), l_e, r_e)
            
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
                k_src_edge, k_des_edge = sp_edge_unique_ascending(k_src_edge, k_des_edge)

            t6 = time.time()
            sp_node_time += t5-t4+(t4-t3)/2
            sp_edge_time += t6-t5+(t4-t3)/2
            del s_e, r_e, l_e, mask1, mask2, sub_edges, u_nbr_ptr

            torch.cuda.empty_cache()

        if k_src_edge.size(0) > 0:
            t7 = time.time()
            link(k_src_edge, k_des_edge, pi)  # 进行连通分量查找
            compress(edge_id, pi)
            src_edge = torch.cat([src_edge, pi[k_src_edge]])
            des_edge = torch.cat([des_edge, k_des_edge])
            src_edge, des_edge = sp_edge_unique_ascending(src_edge, des_edge)
            t8 = time.time()
            equitree_time += (t8-t7)
            # 只更新该k值里合并的节点
            del k_src_edge, k_des_edge
        src_edge = pi[src_edge]
        real_pi[indices[phi_k]] = pi[indices[phi_k]]
        del sub_triangle_count, sub_edge_id, max_nbr_count, group, phi_k
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

    # 去重
    src_edge, des_edge = sp_edge_unique_ascending(src_edge, des_edge)

    # 超节点的映射关系
    node_hash = torch.full((unique_spnode[-1] + 1,), -1, device=device, dtype=torch.int32)
    node_hash[pi_old] = pi
    src_edge = node_hash[src_edge]
    des_edge = node_hash[des_edge]

    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_edge.size(0))

    time_end = time.time()
    print("equitree构造时间：", time_end - time_start,"!!!!")
    print("equitruss构造时间：", time_end - time_start-equitree_time,"!!!!")

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


def run_with_truss(filename: str, name:str=""):
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

