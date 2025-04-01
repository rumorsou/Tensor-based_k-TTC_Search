import random
import time

from CSRGraph4 import read_edge_txt,edgelist_to_CSR
from utils import get_all_nbr, get_all_nbr_cpu, calculate_time, sp_edge_unique_ascending, sp_edge_unique_mask, \
    sp_edge_unique_descending, insert_graph_batch, get_all_nbr_size
import singlegpu_truss
from torch_scatter import segment_csr
from trusstensor import segment_triangle_isin, segment_direct_isin
from maintenance import insert_edge, insert_equitruss, get_affected_edge, get_sub_graph, insert_edge2, get_sub_graph2,\
    get_affected_edge2, get_affected_edge3, get_affected_edge4, get_affected_node_delete
from equitruss5 import equi_truss_construction
# import equitruss_two_step4_6
import equitruss8

import torch

# 保留k-truss等价类，

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
truss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # truss fast better


batch = 20000000  # batch可以调，先控制变量
src_batch = 50000000 # 设置src溢出batch的标准

# batch = 800000  # batch可以调，先控制变量
# src_batch = 50000000 # 设置src溢出batch的标准

@calculate_time
def equi_tree_construction(row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    """
    为了保证truss值低的pi值更小
    """
    t1 = time.time()
    pi = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    
    truss_sort, indices = torch.sort(truss_result, stable=True)
    edge_sort = edge_id[indices]  # indices为int64，作为索引来讲，可以使用
    _, edge_indices = torch.sort(edge_sort, stable = True)  # 按照truss排序后的边再重新排回原位
    edge_indices = edge_indices.to(torch.int32) # edge_indices就指代边在truss排序中的位置
    border_edge_mask = torch.zeros(columns.size(0), device=device, dtype=torch.bool)

    del truss_sort, indices, edge_sort, _

    triangle_count = torch.zeros(columns.size(0), device=device, dtype=torch.int32)

    segment_direct_isin(rows, columns, row_ptr, triangle_count)

    # print("三角形数目:",torch.sum(triangle_count))

    '''
    直接去寻找，所有边的truss值都相等的三角形
    '''
    max_nbr_count = triangle_count.to(torch.int64).cumsum(0)  # 每条边的三角形
    group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch),
                                                           step=batch, dtype=torch.int64, device=device), side='right')
    max_nbr_count = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), max_nbr_count))

    triangles_counts = torch.tensor([], device=device, dtype=torch.int32)
    merge_counts = torch.tensor([], device=device, dtype=torch.int32)

    i = 0
    # 每一波都顶着batch的极限去算
    for head, tail in zip(group[0:-1], group[1:]):
        i+=1

        if head == tail:
            # 说明这个点的邻居太多，很可能难以算完，但是没办法，只能在一次batch中计算
            # 所以continue直到tail后移一位，此时head还是不变，不会影响结果
            continue
        sub_edges = torch.arange(head, tail, device=device, dtype=torch.int32)  # 被选中的边
        s_e = torch.repeat_interleave(sub_edges, triangle_count[sub_edges])
        # u_nbr_indices = max_nbr_count[tail] - max_nbr_count[head]
        u_nbr_ptr = max_nbr_count[head: tail]-max_nbr_count[head]

        l_e = torch.full((s_e.size(0),), -1, device=device, dtype=torch.int32)
        r_e = l_e.clone()
        # 通过subedges获取要处理的边，通过row_ptr获得两边端点，通过unbrptr获得三角形写入的位置
        segment_triangle_isin(rows, columns, row_ptr, sub_edges, u_nbr_ptr.to(torch.int32), l_e, r_e)
        
        triangles_counts = torch.cat((triangles_counts, torch.tensor([s_e.size(0)], device=device, dtype=torch.int32)))
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

        link2(merge_des.flip(0), merge_src.flip(0), pi)


        del s_e, l_e, r_e, s_e2, r_e2, l_e2, mask, mask1, mask2, temp, _, ind, merge_src, merge_des, sub_edges, u_nbr_ptr
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

    del max_nbr_count, group
    compress(edge_id, pi)
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # print(torch.cuda.memory_allocated())

    edge_to_node = torch.full((columns.size(0),), -1, device=device, dtype=torch.long)
    '''
    筛选掉孤立边
    '''
    mask = truss_result != 2
    valid_id = edge_id[mask]  # 【2，3，4，5，6，8】
    pi_old = pi[edge_indices[valid_id]]
    del mask

    # print(torch.cuda.memory_allocated())
    '''
    超节点编号本身以edgeid编号，但edgeid的数量远大于spnode的数量，
    因此进行重新编号
    '''
    # 使用return_inverse即可返回pi中元素在unique中的索引，unique本身排过序了，索引即从0开始重新编号
    # 这时候就已经从小到大开始编号了，以edgeindices为序列的边
    unique_spnode, pi = torch.unique(pi_old, return_inverse=True)

    max_node_id = unique_spnode.size(0) - 1
    edge_to_node[valid_id] = pi  # 这会导致，trussresult小的边spnodeid不一定小

    # print("spnode num", unique_spnode.size(0))
    del unique_spnode, pi_old

    t2 = time.time()
    

    src_edge, des_edge = calucate_super_edges(columns, rows, row_ptr, edge_to_node, edge_id, truss_result, border_edge_mask, edge_indices, triangle_count)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    sp_node_truss = edge_to_node.clone()
    src_edge_truss = src_edge.clone()
    des_edge_truss = des_edge.clone()
    # src, des确实是从小指向大，且truss大的id也大
    # print("spedge num", src_edge.size(0))

    # print("spnode时间:",t2-t1)
    # t3 = time.time()
    # print("spedge时间:",t3-t2)

    # print("开始进行equitree构建")
    node_to_tao = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    tree_pi = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    node_id = tree_pi.clone()
    node_to_tao[pi] = truss_result[valid_id]


    i=0
    while True:
        i+=1
        # print("==tree",i,"==")

        bound = torch.cat((torch.tensor([1], device=device, dtype=torch.int32) ,des_edge[1:] - des_edge[:-1]))
        bound = bound.to(torch.bool)
        selected_indices = torch.where(~bound)[0]
        if selected_indices.size(0)==0:
            break
        des_edge[selected_indices] = src_edge[selected_indices-1]

        des_edge, src_edge = sp_edge_unique_ascending(des_edge, src_edge)

        mask = node_to_tao[des_edge] == node_to_tao[src_edge]
        merge_src = des_edge[mask]
        merge_des = src_edge[mask]

        merge_counts = torch.cat((merge_counts, torch.tensor([merge_src.size(0)],device=device, dtype=torch.int32)))
        # 在这里的link中，merge_src和merge_des从小到大排序，且src>des
        # 这样的数据结构对link更友好
        link(merge_src, merge_des, tree_pi)
        compress(torch.cat((merge_src, merge_des)), tree_pi)

        des_edge = tree_pi[des_edge[~mask]]  # 提前过滤掉(横向)的边
        src_edge = tree_pi[src_edge[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        des_edge, src_edge = sp_edge_unique_descending(des_edge, src_edge)

    compress(node_id, tree_pi)
    unique_spnode, tree_pi = torch.unique(tree_pi, return_inverse=True)
    edge_to_node = tree_pi[edge_to_node]
    max_node_id = unique_spnode.size(0)-1
    des_edge = tree_pi[des_edge]
    src_edge = tree_pi[src_edge]

    t4 = time.time()
    # print("equitree转换时间:",t4-t3)
    # print("spnode num", unique_spnode.size(0))
    # print("spedge num", src_edge.size(0))

    # print("triangles_counts", triangles_counts)
    # print("merge_counts", merge_counts)

    # 返回超节点数据
    return valid_id, edge_to_node.to(torch.int32), src_edge.to(torch.int32), des_edge.to(torch.int32), max_node_id, sp_node_truss.to(torch.int32), src_edge_truss.to(torch.int32), des_edge_truss.to(torch.int32)



def calucate_super_edges(columns: torch.Tensor, rows: torch.Tensor, row_ptr: torch.Tensor, pi: torch.Tensor,
                          edge_id: torch.Tensor, truss_result: torch.Tensor, border_edge_mask: torch.Tensor,
                          edge_indices: torch.Tensor, triangle_count: torch.Tensor ):

    src_edge = torch.tensor([], device=device, dtype=torch.int32)
    des_edge = torch.tensor([], device=device, dtype=torch.int32)

    triangle_count = triangle_count[border_edge_mask]
    sub_edge_id = edge_id[border_edge_mask]
    max_nbr_count = triangle_count.to(torch.int64).cumsum(0)  # 每条边的三角形
    if max_nbr_count.size(0)==0:
        return src_edge, des_edge
    group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch),
                                                           step=batch, dtype=torch.int64, device=device), side='right')
    
    max_nbr_count = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), max_nbr_count))
    # triangles = border_edge_mask.to(torch.int32)
    # triangles[border_edge_mask] = triangle_count
    # max_nbr_count = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), triangles.to(torch.int64).cumsum(0)))
    # torch.cuda.empty_cache()

    i=0
    # 每一波都顶着batch的极限去算
    for head, tail in zip(group[0:-1], group[1:]):
        if head == tail:
            # 说明这个点的邻居太多，很可能难以算完，但是没办法，只能在一次batch中计算
            # 所以continue直到tail后移一位，此时head还是不变，不会影响结果
            continue

        sub_edges = sub_edge_id[head: tail]  # 被选中的边
        s_e = torch.repeat_interleave(sub_edges, triangle_count[head: tail])
        u_nbr_ptr = max_nbr_count[head: tail]-max_nbr_count[head]

        l_e = torch.full((s_e.size(0),), -1, device=device, dtype=torch.int32)
        r_e = l_e.clone()
        # 通过subedges获取要处理的边，通过row_ptr获得两边端点，通过unbrptr获得三角形写入的位置
        segment_triangle_isin(rows, columns, row_ptr, sub_edges, u_nbr_ptr.to(torch.int32), l_e, r_e)

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
        # group_src = torch.cat((s_e2[~mask], s_e2))  # s一定是小节点
        # group_des = torch.cat((l_e2[~mask], r_e2))  # d一定是大节点
        # group_src, group_des = sp_edge_unique2(group_src, group_des)
        src_edge = torch.cat((src_edge, pi[torch.cat((s_e2[~mask], s_e2))]))
        des_edge = torch.cat((des_edge, pi[torch.cat((l_e2[~mask], r_e2))]))
        if src_edge.size(0)>src_batch or tail == group[-1]:
            i+=1
            # print("===去重:",i,"===")
            des_edge, src_edge = sp_edge_unique_descending(des_edge, src_edge)
            # 按照des进行降序
        del s_e, l_e, r_e, s_e2, r_e2, l_e2, mask, mask1, mask2, temp, _, ind
    
    return src_edge, des_edge

def link(e: torch.Tensor, e1: torch.Tensor, pi: torch.Tensor):
    p1 = pi[e]
    p2 = pi[e1]

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

def link2(e: torch.Tensor, e1: torch.Tensor, pi: torch.Tensor):
    p1 = pi[e]
    p2 = pi[e1]
    # mask = p1 != p2
    # print("====")
    i = 0
    # 计算数量
    while p1.size(0) > 0:
        i+=1
        # print(i)
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
        

# 对边或边集进行压缩操作
def compress(e: torch.Tensor, pi: torch.Tensor):
    # 除非已经收敛，否则每条边都不断往上更新到祖先的标签值
    while (pi[pi[e]] == pi[e]).sum().item() != e.shape[0]:
        pi[e] = pi[pi[e]]

def compress2(e: torch.Tensor, pi: torch.Tensor):
    # 除非已经收敛，否则每条边都不断往上更新到祖先的标签值
    while (pi[pi[e]] == pi[e]).sum().item() != e.shape[0]:
        pi[e] = pi[pi[pi[e]]]


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

def eqitree_delete_batch(delete_indices: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor,
                     valid_edge: torch.Tensor, sp_node: torch.Tensor, sp_edge_s: torch.Tensor, sp_edge_e: torch.Tensor):
    row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e, batch_time, free_time = equitruss8.eqitruss_delete_batch(delete_indices, row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e)
    
    print("####truss与tree分界线####")
    time0 = time.time()
    # graph = Graph()
    # graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    # tr = singlegpu_truss.k_truss(graph, 1, row_ptr.clone().shape[0]-1)
    # tr+=2
    # _, _n, _s, _e, _1,_2,_3,_4 = equi_tree_construction(row_ptr, columns, rows, tr)
    # n=torch.unique(_n[_n!=-1])
    # print("spnode num", n.size(0))
    # print("spedge num", _s.size(0)) 
    # del graph, _, _n, _s, _e, _1, _2,_3,_4, n
    # torch.cuda.empty_cache()

    time1 = time.time()
    max_node_id = torch.max(sp_node)
    node_to_tao = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    tree_pi = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    node_id = tree_pi.clone()
    node_to_tao[sp_node[valid_edge]] = truss_result[valid_edge]

    des_edge = sp_edge_e.clone()
    src_edge = sp_edge_s.clone()
    i=0
    while True:
        i+=1
        # print("==tree",i,"==")

        bound = torch.cat((torch.tensor([1], device=device, dtype=torch.int32) ,des_edge[1:] - des_edge[:-1]))
        bound = bound.to(torch.bool)
        selected_indices = torch.where(~bound)[0]
        if selected_indices.size(0)==0:
            break
        des_edge[selected_indices] = src_edge[selected_indices-1]

        des_edge, src_edge = sp_edge_unique_ascending(des_edge, src_edge)

        mask = node_to_tao[des_edge] == node_to_tao[src_edge]
        merge_src = des_edge[mask]
        merge_des = src_edge[mask]

        # 在这里的link中，merge_src和merge_des从小到大排序，且src>des
        # 这样的数据结构对link更友好
        link(merge_src, merge_des, tree_pi)
        compress(torch.cat((merge_src, merge_des)), tree_pi)

        des_edge = tree_pi[des_edge[~mask]]  # 提前过滤掉(横向)的边
        src_edge = tree_pi[src_edge[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        des_edge, src_edge = sp_edge_unique_descending(des_edge, src_edge)

    compress(node_id, tree_pi)
    unique_spnode, tree_pi = torch.unique(tree_pi, return_inverse=True)
    sp_node_tree = tree_pi[sp_node]
    des_edge = tree_pi[des_edge]
    src_edge = tree_pi[src_edge]
    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_edge.size(0))
    time2 = time.time()

    free_time = time1-time0
    batch_time+= time2-time1
    print("***重新构建执行时间===", free_time)
    print("***batch维护执行时间===", batch_time)
    return row_ptr, columns, rows, truss_result, valid_edge, sp_node_tree.to(torch.int32), src_edge.to(torch.int32), des_edge.to(torch.int32), sp_node, sp_edge_s, sp_edge_e


def eqitree_insert_batch(v1: torch.Tensor, v2: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor,
                     valid_edge: torch.Tensor, sp_node: torch.Tensor, sp_edge_s: torch.Tensor, sp_edge_e: torch.Tensor):
    row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e, batch_time, free_time, insert_indices = equitruss8.eqitruss_insert_batch(v1, v2, row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e)
    
    print("truss与tree分界线")
    print(" ")
    time0 = time.time()
    # graph = Graph()
    # graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    # tr = singlegpu_truss.k_truss(graph, 1, row_ptr.clone().shape[0]-1)
    # tr+=2
    # _, _n, _s, _e, _1,_2,_3,_4 = equi_tree_construction(row_ptr, columns, rows, tr)
    # n=torch.unique(_n[_n!=-1])
    # print("spnode num", n.size(0))
    # print("spedge num", _s.size(0)) 
    # del graph, _, _n, _s, _e, _1, _2,_3,_4, n
    # torch.cuda.empty_cache()

    time1 = time.time()

    max_node_id = torch.max(sp_node)
    node_to_tao = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    tree_pi = torch.arange(0, max_node_id+1, device=device, dtype=torch.int32)
    node_id = tree_pi.clone()
    node_to_tao[sp_node[valid_edge]] = truss_result[valid_edge]

    des_edge = sp_edge_e.clone()
    src_edge = sp_edge_s.clone()
    
    # print(torch.where(src_edge>=des_edge)[0]) #并没有满足truss越小，id越小
    # print(torch.where(node_to_tao[src_edge]>=node_to_tao[des_edge])[0])
    i=0
    while True:
        i+=1
        # print("==tree",i,"==")

        bound = torch.cat((torch.tensor([1], device=device, dtype=torch.int32) ,des_edge[1:] - des_edge[:-1]))
        bound = bound.to(torch.bool)
        selected_indices = torch.where(~bound)[0]
        if selected_indices.size(0)==0:
            break
        des_edge[selected_indices] = src_edge[selected_indices-1]

        des_edge, src_edge = sp_edge_unique_ascending(des_edge, src_edge)

        mask = node_to_tao[des_edge] == node_to_tao[src_edge]
        merge_src = des_edge[mask]
        merge_des = src_edge[mask]
        # 在这里的link中，merge_src和merge_des从小到大排序，且src>des
        # 这样的数据结构对link更友好
        link(merge_src, merge_des, tree_pi)
        compress(torch.cat((merge_src, merge_des)), tree_pi)

        des_edge = tree_pi[des_edge[~mask]]  # 提前过滤掉(横向)的边
        src_edge = tree_pi[src_edge[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        des_edge, src_edge = sp_edge_unique_descending(des_edge, src_edge)

    compress(node_id, tree_pi)
    unique_spnode, tree_pi = torch.unique(tree_pi, return_inverse=True)
    sp_node_tree = tree_pi[sp_node]
    des_edge = tree_pi[des_edge]
    src_edge = tree_pi[src_edge]
    print("spnode num", unique_spnode.size(0))
    print("spedge num", src_edge.size(0))
    time2 = time.time()

    free_time = time1-time0
    batch_time+= time2-time1
    print("***重新构建执行时间===", free_time)
    print("***batch维护执行时间===", batch_time)
    return row_ptr, columns, rows, truss_result, valid_edge, sp_node_tree.to(torch.int32), src_edge.to(torch.int32), des_edge.to(torch.int32), sp_node, sp_edge_s, sp_edge_e, insert_indices
    pass

def run(filename: str):
    torch.cuda.empty_cache()
    edge_starts, edge_ends, old_vertices_hash = read_edge_txt(filename, 0)

    row_ptr, columns, rows= edgelist_to_CSR(edge_starts, edge_ends, direct=True)

    row_ptr = torch.tensor(row_ptr, device=truss_device, dtype=torch.int32)
    columns = torch.tensor(columns, device=truss_device, dtype=torch.int32)
    rows = torch.tensor(rows, device=truss_device, dtype=torch.int32)

    graph = Graph()
    graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    truss_result = singlegpu_truss.k_truss(graph, 1, row_ptr.shape[0]-1)
    torch.cuda.synchronize()
    truss_result+=2
    del graph

    row_ptr = torch.tensor(row_ptr, device=truss_device, dtype=torch.int32)
    columns = torch.tensor(columns, device=truss_device, dtype=torch.int32)
    rows = torch.tensor(rows, device=truss_device, dtype=torch.int32)
    truss_result = torch.tensor(truss_result, device=truss_device, dtype=torch.int32)


    print("truss分解完成")
    print("开始equitruss构建")

    # del graph, klist, cnt
    # t1 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())

    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id, sp_node_truss, src_edge_truss, des_edge_truss  = equi_tree_construction(row_ptr, columns, rows, truss_result)
    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id, sp_node_truss, src_edge_truss, des_edge_truss  = equi_tree_construction(row_ptr, columns, rows, truss_result)
    group = [1,10,100,1000,10000]
    # for num in group:
    #     start_v = torch.randint(0, row_ptr.size(0)-2, (num,), device=device, dtype=torch.int32)
    #     end_v = torch.tensor([random.randint(i+1, row_ptr.size(0)-2) for i in start_v], device=device, dtype=torch.int32)

    #     row_ptr, columns, rows, truss_result,valid_edge, sp_node, sp_edge_s, sp_edge_e, sp_node_truss, src_edge_truss, des_edge_truss, insert_indices = eqitree_insert_batch(start_v, end_v, row_ptr, columns, rows, truss_result, valid_edge, sp_node_truss, src_edge_truss, des_edge_truss)
    #     torch.cuda.empty_cache()
    #     print("x=x=x=x=插入删除分界线x=x=x=x=x")
    #     row_ptr, columns, rows, truss_result,valid_edge, sp_node, sp_edge_s, sp_edge_e, sp_node_truss, src_edge_truss, des_edge_truss = eqitree_delete_batch(insert_indices, row_ptr, columns, rows, truss_result, valid_edge, sp_node_truss, src_edge_truss, des_edge_truss)
    #     print("-=-=-=-=下一组别分界线-=-=-=-=")
    #     print(" ")

    # for num in group:
    #     insert_indices = torch.randint(0, columns.size(0)-2, (num,), device=device, dtype=torch.int32)
    #     start_v = rows[insert_indices]
    #     end_v = columns[insert_indices]
    #     row_ptr, columns, rows, truss_result,valid_edge, sp_node, sp_edge_s, sp_edge_e, sp_node_truss, src_edge_truss, des_edge_truss = eqitree_delete_batch(insert_indices, row_ptr, columns, rows, truss_result, valid_edge, sp_node_truss, src_edge_truss, des_edge_truss)
    #     torch.cuda.empty_cache()
    #     print("x=x=x=x=插入删除分界线x=x=x=x=x")
    #     row_ptr, columns, rows, truss_result,valid_edge, sp_node, sp_edge_s, sp_edge_e, sp_node_truss, src_edge_truss, des_edge_truss, insert_indices = eqitree_insert_batch(start_v, end_v, row_ptr, columns, rows, truss_result, valid_edge, sp_node_truss, src_edge_truss, des_edge_truss)
    #     print("-=-=-=-=下一组别分界线-=-=-=-=")
    #     print(" ")

    # edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    # sp_edge_s, sp_edge_e = sp_edge_unique_ascending(sp_edge_s, sp_edge_e)

    # start_v = torch.randint(0, row_ptr.size(0)-2, (100,), device=device, dtype=torch.int32)
    # end_v = torch.tensor([random.randint(i+1, row_ptr.size(0)-1) for i in start_v], device=device, dtype=torch.int32)
    # # start_v = torch.arange(0, 100, device=device, dtype=torch.int32)
    # # end_v = torch.arange(100, 200, device=device, dtype=torch.int32)
    # # start_v = torch.arange(0, 50, device=device, dtype=torch.int32)
    # # end_v = torch.arange(100, 150, device=device, dtype=torch.int32)
    # # # start_v = torch.tensor([3,4,3,1,0,1], device=device, dtype=torch.int32)
    # # # end_v = torch.tensor([8,9,7,8,8,6], device=device, dtype=torch.int32)
    # start_v = torch.tensor([3], device=device, dtype=torch.int32)
    # end_v = torch.tensor([8], device=device, dtype=torch.int32)
    # start_v = torch.tensor([1], device=device, dtype=torch.int32)
    # end_v = torch.tensor([99], device=device, dtype=torch.int32)
    # print(start_v)
    # print(end_v)
    # # delete_indices = torch.arange(0, 50, device=device, dtype=torch.int32)
    # row_ptr, columns, rows, truss_result,valid_edge, sp_node, sp_edge_s, sp_edge_e, sp_node_truss, src_edge_truss, des_edge_truss, insert_indices = eqitree_insert_batch(start_v, end_v, row_ptr, columns, rows, truss_result, valid_edge, sp_node_truss, src_edge_truss, des_edge_truss)
 



if __name__ == '__main__':

    # run("/home/featurize/work/2.18/trussIndexGpu/TETree/equitruss.txt")
    run("/home/featurize/work/2.18/facebook_zero.el")
    # run("/home/featurize/work/2.18/com-amazon_zero.el")
    run("/home/featurize/work/2.18/com-dblp_zero.el")
    # run("/home/featurize/work/2.18/com-youtube_zero.el")
    # run("/home/featurize/work/2.18/soc-catster_zero.el")
    # run("/home/featurize/work/2.18/com-lj_zero.el")
    # run("/home/featurize/work/2.18/orkut_zero.el")
    # run("/home/featurize/work/2.18/weibo.txt")
    # run("/home/featurize/work/2.18/uk-2002.txt")


    # run("/root/autodl-tmp/data/facebook_zero.el")
    # run("/root/autodl-tmp/data/com-amazon_zero.el")
    # run("/root/autodl-tmp/data/com-dblp_zero.el")
    # run("/root/autodl-tmp/data/com-youtube_zero.el")
    # run("/root/autodl-tmp/data/soc-catster_zero.el")
    # run("/root/autodl-tmp/data/com-lj_zero.el")
    # run("/root/autodl-tmp/data/com-orkut.txt")
    # run("/root/autodl-tmp/data/weibo.txt")
    # run("/home/featurize/data/parallel_equitruss-main/path-2-input-graph/weibo.txt")

    # run("/data/code/graph_data/facebook.txt")
    # run("/data/code/graph_data/com-amazon.ungraph.txt")
    # run("/data/code/graph_data/com-dblp.ungraph.txt")
    # run("/data/code/graph_data/com-youtube.ungraph.txt")
    # run("/data/code/graph_data/soc-catster.txt")
    # run("/data/code/graph_data/com-lj.ungraph.txt")
    # run("/data/code/graph_data/com-orkut.ungraph.txt")
    # run("/data/code/graph_data/soc-sinaweibo.mtx")
    
