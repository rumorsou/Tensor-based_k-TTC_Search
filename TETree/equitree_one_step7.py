import random
import time

from CSRGraph4 import read_edge_txt,edgelist_to_CSR
from utils import get_all_nbr, device, sp_edge_unique, calculate_time, cpu, sp_edge_unique2, sp_edge_unique_mask, \
    sp_edge_unique3
# from preprocessing_utils import calucate_triangle_save, calucate_triangle_nosave, calucate_triangle_cpu
# from truss_no_save3 import truss_decomposition
import singlegpu_truss
from truss_class import TrussFile
from torch_scatter import segment_csr
from line_profiler import LineProfiler
from trusstensor import segment_triangle_isin, segment_direct_isin
import equitree_one_step1

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
在equitree9和equitruss_batch5.2的基础上进行，现算三角形。
'''

batch = 20000000
off = 10000000




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
def equi_truss_construction(row_ptr: torch.Tensor, columns: torch.Tensor, truss_result: torch.Tensor):
    """
    """
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

    triangle_count = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
    segment_direct_isin(source_vertices, columns, row_ptr, triangle_count)

    # pi = edge_sort.clone()

    '''
    indices指示按truss排过序的边在pi中的位置
    '''
    _, indices = torch.sort(edge_sort)
    del _

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
        # 每一波都顶着batch的极限去算
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
            del s_e, r_e, l_e, mask1, mask2, sub_edges, u_nbr_ptr

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

    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_edge.size(0))

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


class Graph():
    def __init__(self):
        self.row_ptr = None
        self.columns = None
        self.rows = None
    
    def setup(self, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor):
        self.row_ptr = row_ptr
        self.columns = columns
        self.rows = rows
        self.device = self.columns.device

def run(filename: str):
    edge_starts, edge_ends, old_vertices_hash = read_edge_txt(filename, 0)
    row_ptr, columns, rows= edgelist_to_CSR(edge_starts, edge_ends, direct=True)

    row_ptr = torch.tensor(row_ptr, device=device, dtype=torch.int32)
    columns = torch.tensor(columns, device=device, dtype=torch.int32)
    rows = torch.tensor(rows, device=device, dtype=torch.int32)

    graph = Graph()
    graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    truss_result = singlegpu_truss.k_truss(graph, 1, row_ptr.shape[0]-1)
    torch.cuda.synchronize()
    print(truss_result.size(0))
    print(columns.size(0))
    truss_result+=2
    print(truss_result)

    del graph
    print("truss分解完成")
    print("开始equitruss构建")

    # klist, cnt = torch.unique(truss_result, return_counts=True)
    # print(klist)
    # print(cnt)
    # print("k数量：",klist.size(0))


    # t1 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equitree_one_step1.equi_truss_construction(row_ptr, columns, truss_result)

    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equi_truss_construction(row_ptr, columns, truss_result)


if __name__ == '__main__':


    # run("/home/featurize/work/2.18/facebook_zero.el")
    run("/home/featurize/work/2.18/com-amazon_zero.el")
    # run("/home/featurize/work/2.18/com-dblp_zero.el")
    # run("/home/featurize/work/2.18/com-youtube_zero.el")
    # run("/home/featurize/work/2.18/soc-catster_zero.el")
    # run("/home/featurize/work/2.18/com-lj_zero.el")
    # run("/home/featurize/work/2.18/orkut_zero.el")
    run("/home/featurize/work/2.18/weibo.txt")


# graph:  row_ptr, columns, truss_result
# super_graph:  sp_node, sp_edge_s, sp_edge_e, valid_edge, max_node_id
# pi:  sp_node，边到超节点之间的映射
# 超节点到边的映射
# def query_vertex(v: int, k: int, graph: TrussGraph, super_graph: SuperGraph, sp_node_id, sp_node_truss):
#     # 把tree当成图做，更新truss≥k的索引，使用连通分量
#     # 保证合法的查询
#     if k < 3:
#         print("k<3, 不合法")
#         return
#     if v < 0 or v > graph.max_vertex_id:
#         print("查询节点v不合法")
#         return
#     # 初始化变量
#     visited = torch.full(size=(graph.columns.shape[0],), fill_value=-1, device=device, dtype=torch.long)
#     max_node_id = super_graph.max_node_id
#     src_edge = super_graph.sp_edge_s
#     des_edge = super_graph.sp_edge_e
#     pi = super_graph.sp_node
#     # 更新索引，删除src/des_edge中truss<k的超节点及对应的超边
#     mask = sp_node_truss >= k
#     k_sp_node = sp_node_id[mask]
#     # mask = (torch.isin(src_edge,k_sp_node)) & (torch.isin(des_edge,k_sp_node))
#     mask = src_edge >= k_sp_node[0]
#     src_edge = src_edge[mask]
#     des_edge = des_edge[mask]
#     # 对新索引进行cc算法
#     sub_pi = torch.arange(0, max_node_id + 1, device=device, dtype=torch.long)
#     link(src_edge, des_edge, sub_pi)
#     tmp = torch.arange(0, max_node_id + 1, device=device)
#     compress(tmp, sub_pi)
#     # 对于查询节点v，找到v属于的且truss>k的边
#     mask = torch.zeros(graph.columns.shape[0], device=device, dtype=torch.bool)
#     mask[graph.row_ptr[v]:graph.row_ptr[v + 1]] = True
#     mask = mask | (graph.columns == v)
#     # vq所在的超节点
#     v_sp_node = torch.unique(pi[mask])
#     v_sp_node = v_sp_node[v_sp_node != -1]
#     v_sp_node_pi = sp_node_truss[v_sp_node]
#     mask = v_sp_node_pi >= k
#     v_sp_node = v_sp_node[mask]
#     # 找到对应的超节点和社区编号
#     community_id = torch.unique(sub_pi[v_sp_node])
#     mask = torch.isin(sub_pi, community_id)
#     all_sp_node = sp_node_id[mask]
#     # 给graph的边分配社区编号
#     mask = torch.isin(pi, all_sp_node)
#     visited[mask] = sub_pi[pi[mask]]
#     return visited


# def query_vertex_2(v: int, k: int, graph: TrussGraph, super_graph: SuperGraph, sp_node_id, sp_node_truss, ptr,
#                    sorted_pi, idx, sp_ptr):
#     # 根据树的结构，传递root的id到整个子树
#     # 保证合法的查询
#     if k < 3:
#         print("k<3, 不合法")
#         return
#     if v < 0 or v > graph.max_vertex_id:
#         print("查询节点v不合法")
#         return
#     # 初始化变量
#     visited = torch.full(size=(graph.columns.shape[0],), fill_value=-1, device=device, dtype=torch.long)
#     max_node_id = super_graph.max_node_id
#     src_edge = super_graph.sp_edge_s
#     des_edge = super_graph.sp_edge_e
#     pi = super_graph.sp_node
#     root = torch.full(size=(max_node_id + 1,), fill_value=-1, device=device, dtype=torch.long)
#     root[sp_node_id] = sp_node_id
#     # 传递root的id，给子树赋根节点的id
#     ptr = ptr[k:]
#     while ptr.shape[0] != 1:
#         root[des_edge[ptr[0]:ptr[1]]] = root[src_edge[ptr[0]:ptr[1]]]
#         ptr = ptr[1:]
#     # 对于查询节点v，找到v属于的且truss>k的边
#     mask = torch.zeros(graph.columns.shape[0], device=device, dtype=torch.bool)
#     mask[graph.row_ptr[v]:graph.row_ptr[v + 1]] = True
#     mask = mask | (graph.columns == v)
#     # 找到v所在的truss>k的超节点
#     v_sp_node = torch.unique(pi[mask])
#     v_sp_node = v_sp_node[v_sp_node != -1]
#     v_sp_node_pi = sp_node_truss[v_sp_node]
#     mask = v_sp_node_pi >= k
#     v_sp_node = v_sp_node[mask]
#     # 找到vq超节点的根节点
#     sp_node_root = torch.unique(root[v_sp_node])
#     # 找到属于这些根节点的全部超节点
#     sorted_root, idxs = torch.sort(root)
#     _, cnt = torch.unique_consecutive(sorted_root, return_counts=True)
#     ptr = torch.cumsum(torch.cat((torch.tensor([0], device=device, dtype=cnt.dtype), cnt), dim=0), dim=0)
#     start = sp_node_root
#     end = sp_node_root + 1
#     mask = get_all_nbr(ptr[start], ptr[end])
#     all_sp_node = idxs[mask]

#     start = all_sp_node
#     end = all_sp_node + 1
#     mask = get_all_nbr(sp_ptr[start], sp_ptr[end])
#     visited[idx[mask]] = root[sorted_pi[mask]]
#     return visited


# def query_vertex_3(v: int, k: int, graph: TrussGraph, super_graph: SuperGraph, sp_node_id, sp_node_truss, ptr,
#                    sorted_pi, idx, sp_ptr):
#     # 修改后的root，各节点先指向自己，由des指向src，再compress
#     # 保证合法的查询
#     if k < 3:
#         print("k<3, 不合法")
#         return
#     if v < 0 or v > graph.max_vertex_id:
#         print("查询节点v不合法")
#         return
#     # 初始化变量
#     visited = torch.full(size=(graph.columns.shape[0],), fill_value=-1, device=device, dtype=torch.long)
#     max_node_id = super_graph.max_node_id
#     src_edge = super_graph.sp_edge_s
#     des_edge = super_graph.sp_edge_e
#     pi = super_graph.sp_node
#     root = torch.full(size=(max_node_id + 1,), fill_value=-1, device=device, dtype=torch.long)
#     # 更新索引，删除src/des_edge中truss<k的超节点及对应的超边
#     mask = sp_node_truss >= k
#     k_sp_node = sp_node_id[mask]
#     mask = src_edge >= k_sp_node[0]
#     src_edge = src_edge[mask]
#     des_edge = des_edge[mask]
#     # 初始化root，使其指向自己,再让des_edge的root指向src_edge
#     root[sp_node_id] = sp_node_id
#     root[des_edge] = src_edge
#     # compress
#     compress(sp_node_id, root)
#     # 对于查询节点v，找到v属于的且truss>k的边
#     mask = torch.zeros(graph.columns.shape[0], device=device, dtype=torch.bool)
#     mask[graph.row_ptr[v]:graph.row_ptr[v + 1]] = True
#     mask = mask | (graph.columns == v)
#     # 找到v所在的truss>k的超节点
#     v_sp_node = torch.unique(pi[mask])
#     v_sp_node = v_sp_node[v_sp_node != -1]
#     v_sp_node_pi = sp_node_truss[v_sp_node]
#     mask = v_sp_node_pi >= k
#     v_sp_node = v_sp_node[mask]

#     # 找到vq超节点的根节点
#     sp_node_root = torch.unique(root[v_sp_node])
#     # 找到属于这些根节点的全部超节点
#     # mask = torch.isin(root,sp_node_root)
#     # all_sp_node = sp_node_id[mask]
#     sorted_root, idxs = torch.sort(root)
#     _, cnt = torch.unique_consecutive(sorted_root, return_counts=True)
#     ptr = torch.cumsum(torch.cat((torch.tensor([0], device=device, dtype=cnt.dtype), cnt), dim=0), dim=0)
#     start = sp_node_root
#     end = sp_node_root + 1
#     mask = get_all_nbr(ptr[start], ptr[end])
#     all_sp_node = idxs[mask]

#     # mask = torch.isin(pi,all_sp_node)
#     # visited[mask] = root[pi[mask]]
#     start = all_sp_node
#     end = all_sp_node + 1
#     mask = get_all_nbr(sp_ptr[start], sp_ptr[end])
#     visited[idx[mask]] = root[sorted_pi[mask]]

#     return visited


# 对有序的求交集
def intersection(values, boundaries):  # value和mask都有序
    mask = values <= boundaries[-1]  # 这个是顺序的，应该可以再次加速的
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result] == values
    return mask


# def run(filename: str):
#     edge_starts, edge_ends, old_vertices_hash = read_edge_txt(filename, 0)
#     row_ptr, columns, source_vertices = edgelist_to_CSR(edge_starts, edge_ends, direct=True)

#     row_ptr = torch.tensor(row_ptr, device=device, dtype=torch.int32)
#     columns = torch.tensor(columns, device=device, dtype=torch.int32)
#     source_vertices = torch.tensor(source_vertices, device=device, dtype=torch.int32)
#     del edge_starts, edge_ends
#     print("finish...")

#     # 边和三角形太多，无法用存三角形的truss分解了
#     truss_result, edge_supports = truss_decomposition(row_ptr, columns)
#     print("truss分解完成")
#     print("开始equitruss构建")

#     # t1 = time.time()
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()

#     edge_supports2 = edge_supports.clone()
#     columns2 = columns.clone()

#     valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equi_truss_construction(row_ptr, columns, truss_result)





# if __name__ == '__main__':
#     run("../../../graph_data/com-youtube.ungraph.txt")
    # run(1, "experiment_data/com-dblp.ungraph.txt")
    # run(1, "experiment_data/com-youtube.ungraph.txt")
    # run(2, "catster_no.pt")
    # run(2, "comlj_no.pt")
