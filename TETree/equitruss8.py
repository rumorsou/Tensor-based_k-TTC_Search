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
# import equitruss_two_step4_6

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
truss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # truss fast better


# 

batch = 20000000  # batch可以调，先控制变量
src_batch = 50000000 # 设置src溢出batch的标准

@calculate_time
def equi_truss_construction(row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
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

    edge_to_node = torch.full((columns.size(0),), -1, device=device, dtype=torch.int32)
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
    edge_to_node[valid_id] = pi.to(torch.int32)
    # print("spnode num", unique_spnode.size(0))
    del unique_spnode, pi_old

    t2 = time.time()
    

    src_edge, des_edge = calucate_super_edges(columns, rows, row_ptr, edge_to_node, edge_id, truss_result, border_edge_mask, edge_indices, triangle_count)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())

    # print("spedge num", src_edge.size(0))

    # print("spnode时间:",t2-t1)
    # t3 = time.time()
    # print("spedge时间:",t3-t2)

    return valid_id, edge_to_node, src_edge.to(torch.int32), des_edge.to(torch.int32), max_node_id



def calucate_super_edges(columns: torch.Tensor, rows: torch.Tensor, row_ptr: torch.Tensor, pi: torch.Tensor,
                          edge_id: torch.Tensor, truss_result: torch.Tensor, border_edge_mask: torch.Tensor,
                          edge_indices: torch.Tensor, triangle_count: torch.Tensor ):

    src_edge = torch.tensor([], device=device, dtype=torch.int32)
    des_edge = torch.tensor([], device=device, dtype=torch.int32)

    triangle_count = triangle_count[border_edge_mask]
    sub_edge_id = edge_id[border_edge_mask]
    max_nbr_count = triangle_count.to(torch.int64).cumsum(0)  # 每条边的三角形
    if max_nbr_count.size(0)<=0:
        return src_edge, des_edge
    group = torch.searchsorted(max_nbr_count, torch.arange(0, int(max_nbr_count[-1] + batch),
                                                           step=batch, dtype=torch.int64, device=device), side='right')
    
    max_nbr_count = torch.cat((torch.zeros(1, device=device, dtype=torch.int64), max_nbr_count))

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
        src_edge = torch.cat((src_edge, pi[torch.cat((s_e2[~mask], s_e2))]))
        des_edge = torch.cat((des_edge, pi[torch.cat((l_e2[~mask], r_e2))]))
        if src_edge.size(0)>src_batch or tail == group[-1]:
            i+=1
            # print("===去重:",i,"===")
            des_edge, src_edge = sp_edge_unique_descending(des_edge, src_edge)
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
        # print(h)
        # print(l)
        # print("===")

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

def link3(e: torch.Tensor, e1: torch.Tensor, pi: torch.Tensor, node_to_edge: torch.Tensor):
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
        h_edge = node_to_edge[h]

        # 判断是否已收敛到祖先
        mask1 = pi[h_edge] == h

        # 重点在这，祖先可能被变化多次，只会以最后一次为主
        pi[h_edge[mask1]] = l[mask1]  # 是则将祖先指向较小的那个

        # 被变化多次的祖先中，只筛选已经相等并收敛的
        mask2 = (pi[h_edge] == l) & mask1

        h_edge = h_edge[~mask2]
        l = l[~mask2]

        p2 = pi[node_to_edge[pi[h_edge]]]
        p1 = pi[node_to_edge[l]]
        

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

# def eqitruss_delete_batch(v1: torch.Tensor, v2: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor,
#                      valid_edge: torch.Tensor, sp_node: torch.Tensor, sp_edge_s: torch.Tensor, sp_edge_e: torch.Tensor, max_node_id: torch.Tensor, edge_id: torch.Tensor):
def eqitruss_delete_batch(delete_indices: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor,
                     valid_edge: torch.Tensor, sp_node: torch.Tensor, sp_edge_s: torch.Tensor, sp_edge_e: torch.Tensor):
    '''
    测试时保证输入的边的两个端点均存在，实际需要加上判断条件
    '''
    time1 = time.time()
    # off = 100000000
    # edge_uid = rows*off+columns
    # insert_edge_uid = v1*off+v2
    # delete_mask = torch.isin(edge_uid, insert_edge_uid)
    # del_edge_id = torch.where(delete_mask)[0]
    delete_mask = torch.zeros(columns.size(0), device=device, dtype=torch.bool)
    del_edge_id = delete_indices
    delete_mask[del_edge_id]=True

    max_node_id = torch.max(sp_node)
    del_node = torch.unique(sp_node[[del_edge_id]])
    del_node = del_node[del_node!=-1]
    # print("被删除的超节点==",del_node)
    if del_node.size(0)<=0:
        rows = rows[~delete_mask]
        columns = columns[~delete_mask]
        count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
        vertices, cnt = torch.unique_consecutive(rows, return_counts = True)
        count[vertices] = cnt.to(torch.int32)
        row_ptr = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32)))
        edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

        # truss_result和sp_node为新插入的边预留位置
        new_tensor = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
        truss_result=truss_result[~delete_mask]
        sp_node = sp_node[~delete_mask]
        # new_tensor = truss_result[~delete_mask]
        # truss_result = new_tensor.clone()
        # new_tensor = sp_node[~delete_mask]
        # sp_node = new_tensor.clone()

        # mask1 = torch.isin(sp_edge_s, delete_node)  
        # mask2 = torch.isin(sp_edge_e, delete_node) 
        # mask = mask1 | mask2  # 去除被删除的超节点
        # sp_edge_s = sp_edge_s[~mask]
        # sp_edge_e = sp_edge_e[~mask]
        timex = time.time()
        valid_edge = edge_id[truss_result>2]
        print("***batch维护执行时间===", timex-time1)
        print("提前终止！！！1")
        return row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s.to(torch.int32), sp_edge_e.to(torch.int32), 0, 0
    node_truss = torch.zeros(max_node_id+2, device=device, dtype=torch.int32)
    node_truss[sp_node] = truss_result
    count = torch.zeros(max_node_id+1, device=device, dtype=torch.int32)
    unique_node, cnt=torch.unique(sp_node[valid_edge], return_counts=True)
    count[unique_node] = cnt.to(torch.int32)
    sp_row_ptr =  torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32)))
    aff_node = get_affected_node_delete(del_node, sp_row_ptr, sp_edge_e, sp_edge_s, node_truss)
    if aff_node.shape[0]>0:
        aff_node = torch.unique(torch.cat((aff_node, del_node)))
    else:
        aff_node = del_node
    # print("affnode==", aff_node)

    rows = rows[~delete_mask]
    columns = columns[~delete_mask]
    count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32)))
    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    # print(rows.size(0))
    # print(columns.size(0))
    # print(row_ptr[-1])
    # # truss_result和sp_node为新插入的边预留位置
    # print(sp_node.size(0))
    truss_result=truss_result[~delete_mask]
    sp_node = sp_node[~delete_mask]
    # print(sp_node)
    # print(sp_node.size(0))
    # new_tensor = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
    # new_tensor = truss_result[~delete_mask]
    # truss_result = new_tensor.clone()
    # new_tensor = sp_node[~delete_mask]
    # sp_node = new_tensor.clone()

    # print("CSR调整完成")

    time2 = time.time()
    # graph = Graph()
    # graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    # tr = singlegpu_truss.k_truss(graph, 1, row_ptr.clone().shape[0]-1)
    # tr+=2
    # _, _n, _s, _e, _1 = equi_truss_construction(row_ptr, columns, rows, tr)
    # n=torch.unique(_n[_n!=-1])
    # print("spnode num", n.size(0))
    # print("spedge num", _s.size(0)) 
    # del graph, _, _n, _s, _e, _1, n
    # torch.cuda.empty_cache()
    # # mask = truss_result!=tr
    # # print("mask===", torch.where(mask)[0])
    # # print("==========+++++++++")
    # print(" ")
    time3 = time.time()  


    mask1 = torch.isin(sp_edge_s, aff_node)  #  一定有affnode，不一定有affedge
    mask2 = torch.isin(sp_edge_e, aff_node) # 与内部超节点有关的超边全部删除
    mask = mask1 | mask2  # 去除被删除的超节点
    sp_edge_s = sp_edge_s[~mask]
    sp_edge_e = sp_edge_e[~mask]
    # print("spedge数量：", sp_edge_s.size(0))

    aff_edge_mask = torch.isin(sp_node, aff_node)
    aff_edge = edge_id[aff_edge_mask]

    # print("aff_edge===", aff_edge)
    if aff_edge.size(0)<=0:
        valid_edge = edge_id[truss_result>2]
        print("提前终止！！！2")
        print("***重新构建执行时间===", time3-time2)
        print("***batch维护执行时间===", time2-time1)
        return row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s.to(torch.int32), sp_edge_e.to(torch.int32), time2-time1, time3-time2
    sub_edge, border_edge, sub_row_ptr, sub_columns, sub_rows = get_sub_graph(aff_edge, row_ptr, columns, rows, edge_id)
    # print("子图边数量和边界边数量===", sub_edge.size(0), border_edge.size(0))

    sub_edge_id = torch.arange(0, sub_columns.size(0), device=device, dtype=torch.int32)
    # hash是从sub_edge映射到sub_edge_id；反过来则无需映射
    sub_edge_hash = torch.full((sub_edge[-1] + 1,), -1, device=device, dtype=torch.int32)
    sub_edge_hash[sub_edge] = sub_edge_id
    h_edge, mask_sub, h_row_ptr, h_columns, h_rows = get_sub_graph2(sub_edge, row_ptr, columns, rows, edge_id) # 确保border_edge的三角形和truss结果的准确性

    print("truss分解数据量===",h_edge.size(0))

    graph = Graph()
    graph.setup(h_row_ptr.clone(), h_columns.clone(), h_rows.clone())
    sub_truss_result = singlegpu_truss.k_truss(graph, 1, sub_row_ptr.shape[0]-1)
    sub_truss_result = sub_truss_result[mask_sub] # 只提取subgraph的truss结果
    torch.cuda.synchronize()
    sub_truss_result+=2



    sub_truss_result[sub_edge_hash[border_edge]] = truss_result[border_edge]
    sub_valid_edge, sub_sp_node, sub_sp_edge_s, sub_sp_edge_e, sub_max_node_id  = equi_truss_construction(sub_row_ptr, sub_columns, sub_rows, sub_truss_result)
    # sub_valid_edge, sub_sp_node, sub_sp_edge_s, sub_sp_edge_e, sub_max_node_id  = equi_truss_construction(h_row_ptr, h_columns, h_rows, sub_truss_result)


    # print("sub计算完成")

    truss_result[sub_edge] = sub_truss_result  # 只更新受影响边的truss值
    # mask = truss_result!=tr
    # print("trussness维护是否正确===", torch.where(mask)[0])
    # 超节点重编号
    max_origin_node = torch.max(sp_node) + 1
    # 这样就不会重复了(只让合法的边的超节点重编号，不合法的边仍未-1)
    sub_sp_node[sub_valid_edge] += max_origin_node
    sub_sp_edge_s += max_origin_node
    sub_sp_edge_e += max_origin_node

    origin_border_edge_node = sp_node[border_edge]
    border_edge_node = sub_sp_node[sub_edge_hash[border_edge]]
    mask = (origin_border_edge_node!=-1) | (border_edge_node!=-1)
    origin_border_edge_node = origin_border_edge_node[mask]
    border_edge_node = border_edge_node[mask]
    # sp_node[sub_edge] = sub_sp_node 

    temp_sp_node =torch.cat((sp_node, sub_sp_node))
    max_node_id = torch.max(temp_sp_node)
    pi = torch.full((max_node_id+2, ), -1, device=device, dtype=torch.int32) #最后一个也是-1
    pi[temp_sp_node] = temp_sp_node


    node_truss = torch.zeros(max_node_id+1, device=device, dtype=torch.int32)
    mask = sp_node!=-1
    node_truss[sp_node[mask]] = truss_result[edge_id[mask]]
    mask = sub_sp_node!=-1
    node_truss[sub_sp_node[mask]] = truss_result[sub_edge[mask]]


    mask = origin_border_edge_node > border_edge_node
    high_node = torch.where(mask, origin_border_edge_node, border_edge_node)  # 大标签
    low_node = origin_border_edge_node + border_edge_node - high_node  # 小标签
   
    sp_edge_s = torch.cat((sp_edge_s,sub_sp_edge_s, low_node))
    sp_edge_e = torch.cat((sp_edge_e, sub_sp_edge_e, high_node))

    while True:
        mask = node_truss[sp_edge_s] == node_truss[sp_edge_e]
        if torch.where(mask)[0].size(0)<=0:
            break
        merge_src = sp_edge_s[mask]
        merge_des = sp_edge_e[mask]
        link(merge_src, merge_des, pi)
        compress(torch.cat((merge_src, merge_des)), pi)

        sp_edge_s = pi[sp_edge_s[~mask]]  # 提前过滤掉(横向)的边
        sp_edge_e = pi[sp_edge_e[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        sp_edge_e, sp_edge_s = sp_edge_unique_descending(sp_edge_e, sp_edge_s)
    
    sp_node = pi[sp_node]
    sub_sp_node = pi[sub_sp_node]
    sp_node[aff_edge] = sub_sp_node[sub_edge_hash[aff_edge]]
    # print(sp_node)
    # print(sp_edge_s)
    # print(sp_edge_e)

    # 重编号
    valid_mask = truss_result>2
    valid_edge = edge_id[valid_mask]
    off = 100000000
    new_sp_node = truss_result[valid_mask].to(torch.long)*off+sp_node[valid_mask].to(torch.long)  # 保证了truss值越大，nodeid越大，但是一个truss值相等的id则不由边id顺序排列，是否会出现问题
    _, new_sp_node= torch.unique(new_sp_node, return_inverse=True)
    # sp_node_to_new = torch.zeros(torch.max(sp_node)+2, device=device, dtype=torch.int32)
    pi[sp_node[valid_mask]] = new_sp_node.to(torch.int32)
    sp_node[valid_mask] = new_sp_node.to(torch.int32)
    sp_edge_s = pi[sp_edge_s] # 由超节点转为新的超节点
    sp_edge_e = pi[sp_edge_e]
    sp_edge_e, sp_edge_s = sp_edge_unique_descending(sp_edge_e, sp_edge_s) # 重新降序排列

    # max_node_id = int(torch.max(sp_node[valid_edge]))
    # print("=======索引维护结果=======")
    unique_spnode = torch.unique(sp_node[valid_edge])
    print("spnode num", unique_spnode.size(0))
    print("spedge num", sp_edge_s.size(0))    

    time4 = time.time()
    free_time = time3-time2
    batch_time =  time4-time1-(time3-time2)
    once_time = time4-time3

    print("***重新构建执行时间===", free_time)
    print("***batch维护执行时间===", batch_time)
    return row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s.to(torch.int32), sp_edge_e.to(torch.int32), batch_time, free_time


def eqitruss_insert_batch(v1: torch.Tensor, v2: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor,
                     valid_edge: torch.Tensor, sp_node: torch.Tensor, sp_edge_s: torch.Tensor, sp_edge_e: torch.Tensor):
    '''
    测试时保证输入的边的两个端点均存在，实际需要加上判断条件
    '''
    time1 = time.time()
    rows = torch.cat((rows, v1))
    columns = torch.cat((columns, v2))
    insert_mask = torch.zeros(columns.size(0), device=device, dtype=torch.bool)
    insert_mask[truss_result.size(0):] = True
    # 更新CSR图结构
    rows, columns, insert_mask = insert_graph_batch(rows, columns, insert_mask) # 插入的边有可能存在

    if torch.where(insert_mask)[0].size(0)==0:
        timex = time.time()
        # 如果插入的都是已经存在的边，则直接返回
        print("***batch维护执行时间===", timex-time1)
        print("提前终止！！！")
        return row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s.to(torch.int32), sp_edge_e.to(torch.int32), 0, 0
    # print("insert_edge===", torch.where(insert_mask)[0])
    # print("insert_mask:", torch.where(insert_mask)[0].size(0))

    count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32)))
    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    # truss_result和sp_node为新插入的边预留位置
    new_tensor = torch.zeros(columns.size(0), device=device, dtype=torch.int32)
    new_tensor[~insert_mask] = truss_result
    truss_result = new_tensor.clone()
    new_tensor[~insert_mask] = sp_node
    new_tensor[insert_mask] = -1
    sp_node = new_tensor.clone()

    '''
    对于超大数据集，记录一下最初truss分解的时间和构建的时间即可。大致没有太大区别
    '''
    time2 = time.time()
    # graph = Graph()
    # graph.setup(row_ptr.clone(), columns.clone(), rows.clone())
    # tr = singlegpu_truss.k_truss(graph, 1, row_ptr.clone().shape[0]-1)
    # tr+=2
    # _, _n, _s, _e, _1 = equi_truss_construction(row_ptr, columns, rows, tr)
    # n=torch.unique(_n[_n!=-1])
    # print("spnode num", n.size(0))
    # print("spedge num", _s.size(0)) 
    # del graph, _, _n, _s, _e, _1, n
    # torch.cuda.empty_cache()
    # mask = truss_result!=tr
    # print("mask===", torch.where(mask)[0])
    # print("==========+++++++++")
    print(" ")
    time3 = time.time()  

    aff_edge = get_affected_edge2(insert_mask, row_ptr, columns, rows, truss_result) #以插入边的三角形个数作为上界
    # print("初始的affedge===", aff_edge)
    aff_node = torch.unique(sp_node[aff_edge])  # affnode为空
    size1 = int(aff_node.size(0))
    aff_node = aff_node[aff_node!=-1]
    # print("是否存在为-1的affnode", int(aff_node.size(0))-size1)
    aff_edge_mask = torch.isin(sp_node, aff_node)
    # print(torch.where(insert_mask)[0])
    aff_edge_mask[insert_mask] = True
    aff_edge = torch.unique(torch.cat((edge_id[aff_edge_mask], aff_edge)))
    # print("affedge===", aff_edge)
    # print(aff_edge)

    sub_edge, border_edge, sub_row_ptr, sub_columns, sub_rows = get_sub_graph(aff_edge, row_ptr, columns, rows, edge_id)
    # print("子图边数量和边界边数量===", sub_edge.size(0), border_edge.size(0))

    sub_edge_id = torch.arange(0, sub_columns.size(0), device=device, dtype=torch.int32)
    # hash是从sub_edge映射到sub_edge_id；反过来则无需映射
    sub_edge_hash = torch.full((sub_edge[-1] + 1,), -1, device=device, dtype=torch.int32)
    sub_edge_hash[sub_edge] = sub_edge_id
    h_edge, mask_sub, h_row_ptr, h_columns, h_rows = get_sub_graph2(sub_edge, row_ptr, columns, rows, edge_id) # 确保border_edge的三角形准确性


    print("truss分解数据量===",h_columns.size(0))

    graph = Graph()
    graph.setup(h_row_ptr.clone(), h_columns.clone(), h_rows.clone())
    sub_truss_result = singlegpu_truss.k_truss(graph, 1, h_row_ptr.shape[0]-1)
    sub_truss_result = sub_truss_result[mask_sub] # 只提取subgraph的truss结果
    torch.cuda.synchronize()
    sub_truss_result+=2

    sub_truss_result[sub_edge_hash[border_edge]] = truss_result[border_edge]
    truss_result2 = truss_result.clone()
    truss_result2[sub_edge] = sub_truss_result
    mask = truss_result!=truss_result2
    aff_edge = edge_id[mask] 
    # 只有这些边的trussness发生变化，那么这些边的超节点需要率先调整
    aff_node = torch.unique(sp_node[aff_edge])  # affnode为空
    size1 = int(aff_node.size(0))
    aff_node = aff_node[aff_node!=-1]
    # print("是否存在为-1的affnode", int(aff_node.size(0))-size1)
    aff_edge_mask = torch.isin(sp_node, aff_node)
    aff_edge_mask[insert_mask] = True
    aff_edge = torch.unique(torch.cat((edge_id[aff_edge_mask], aff_edge)))

    truss_result = truss_result2
    # mask = truss_result!=tr
    # print("trussness维护是否正确===", torch.where(mask)[0])
    if aff_node.shape[0]>0: # 对于插入情况，一定有aff_edge，不一定有aff_node
        mask1 = torch.isin(sp_edge_s, aff_node)
        mask2 = torch.isin(sp_edge_e, aff_node) # 与内部超节点有关的超边全部删除
        mask = mask1 | mask2  # 去除被删除的超节点
        sp_edge_s = sp_edge_s[~mask]
        sp_edge_e = sp_edge_e[~mask]
    # print("spedge数量：", sp_edge_s.size(0))

    sub_edge, border_edge, sub_row_ptr, sub_columns, sub_rows = get_sub_graph(aff_edge, row_ptr, columns, rows, edge_id)
    sub_truss_result = truss_result2[sub_edge]
    sub_edge_id = torch.arange(0, sub_columns.size(0), device=device, dtype=torch.int32)
    sub_edge_hash = torch.full((sub_edge[-1] + 1,), -1, device=device, dtype=torch.int32)
    sub_edge_hash[sub_edge] = sub_edge_id

    sub_valid_edge, sub_sp_node, sub_sp_edge_s, sub_sp_edge_e, sub_max_node_id  = equi_truss_construction(sub_row_ptr, sub_columns, sub_rows, sub_truss_result)

    ## 这里得到的子图spnodeid也跟truss值有关
    # print("sub计算完成")

    # truss_result[sub_edge] = sub_truss_result  # 只更新受影响边的truss值
    # mask = truss_result!=tr
    # print("trussness维护是否正确===", torch.where(mask)[0])
    # 超节点重编号
    max_origin_node = torch.max(sp_node) + 1
    # 这样就不会重复了(只让合法的边的超节点重编号，不合法的边仍未-1)
    sub_sp_node[sub_valid_edge] += max_origin_node
    sub_sp_edge_s += max_origin_node
    sub_sp_edge_e += max_origin_node

    origin_border_edge_node = sp_node[border_edge]
    border_edge_node = sub_sp_node[sub_edge_hash[border_edge]]
    mask = (origin_border_edge_node!=-1) | (border_edge_node!=-1)
    origin_border_edge_node = origin_border_edge_node[mask]
    border_edge_node = border_edge_node[mask]
    # sp_node[sub_edge] = sub_sp_node 

    temp_sp_node =torch.cat((sp_node, sub_sp_node))
    max_node_id = torch.max(temp_sp_node)
    pi = torch.full((max_node_id+2, ), -1, device=device, dtype=torch.int32) #最后一个也是-1
    pi[temp_sp_node] = temp_sp_node


    node_truss = torch.zeros(max_node_id+1, device=device, dtype=torch.int32)
    mask = sp_node!=-1
    node_truss[sp_node[mask]] = truss_result[edge_id[mask]]
    mask = sub_sp_node!=-1
    node_truss[sub_sp_node[mask]] = truss_result[sub_edge[mask]]


    mask = origin_border_edge_node > border_edge_node
    high_node = torch.where(mask, origin_border_edge_node, border_edge_node)  # 大标签
    low_node = origin_border_edge_node + border_edge_node - high_node  # 小标签
   
    sp_edge_s = torch.cat((sp_edge_s,sub_sp_edge_s, low_node))
    sp_edge_e = torch.cat((sp_edge_e, sub_sp_edge_e, high_node))

    while True:
        mask = node_truss[sp_edge_s] == node_truss[sp_edge_e]
        if torch.where(mask)[0].size(0)<=0:
            break
        merge_src = sp_edge_s[mask]
        merge_des = sp_edge_e[mask]
        link(merge_src, merge_des, pi)
        compress(torch.cat((merge_src, merge_des)), pi)

        sp_edge_s = pi[sp_edge_s[~mask]]  # 提前过滤掉(横向)的边
        sp_edge_e = pi[sp_edge_e[~mask]]
        # mask = src_node != des_node
        # 结果需要按src降序排列
        sp_edge_e, sp_edge_s = sp_edge_unique_descending(sp_edge_e, sp_edge_s)
    
    sp_node = pi[sp_node]
    sub_sp_node = pi[sub_sp_node]
    sp_node[aff_edge] = sub_sp_node[sub_edge_hash[aff_edge]]
    # print(sp_node)
    # print(sp_edge_s)
    # print(sp_edge_e)

    valid_mask = truss_result>2
    valid_edge = edge_id[valid_mask]
    off = 100000000
    new_sp_node = truss_result[valid_mask].to(torch.long)*off+sp_node[valid_mask].to(torch.long)  # 保证了truss值越大，nodeid越大，但是一个truss值相等的id则不由边id顺序排列，是否会出现问题
    _, new_sp_node= torch.unique(new_sp_node, return_inverse=True)
    # sp_node_to_new = torch.zeros(torch.max(sp_node)+2, device=device, dtype=torch.int32)
    pi[sp_node[valid_mask]] = new_sp_node.to(torch.int32)
    sp_node[valid_mask] = new_sp_node.to(torch.int32)
    sp_edge_s = pi[sp_edge_s] # 由超节点转为新的超节点
    sp_edge_e = pi[sp_edge_e]
    # print(torch.where(sp_edge_s>sp_edge_e)[0])
    # print(torch.where(sp_edge_s==sp_edge_e)[0])
    # mask = torch.where(sp_edge_s>sp_edge_e)[0]
    # print(sp_edge_s[mask])
    # print(sp_edge_e[mask])
    sp_edge_e, sp_edge_s = sp_edge_unique_descending(sp_edge_e, sp_edge_s) # 重新降序排列
    # _, ind=torch.sort(truss_result)
    # sp_node=sp_node[ind]
    # print(torch.where(torch.diff(sp_node[valid_mask])<0)[0])
    # 重编号
    # mask = truss_result > 2
    # valid_edge = edge_id[mask]
    # _, sp_node_2 = torch.unique(sp_node[mask], return_inverse=True)
    # sp_node_2 = sp_node_2.to(torch.int32)
    # pi[sp_node[mask]] = sp_node_2
    # sp_edge_s = pi[sp_edge_s]
    # sp_edge_e = pi[sp_edge_e]
    # sp_node[mask]=sp_node_2


    # max_node_id = int(torch.max(sp_node[valid_edge]))
    # print("=======索引维护结果=======")
    unique_spnode = torch.unique(sp_node[valid_edge])
    print("spnode num", unique_spnode.size(0))
    print("spedge num", sp_edge_s.size(0))    

    time4 = time.time()
    free_time = time3-time2
    batch_time =  time4-time1-(time3-time2)
    once_time = time4-time3

    print("***重新构建执行时间===", free_time)
    print("***batch维护执行时间===", batch_time)
    insert_indices = torch.where(insert_mask)[0]
    return row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s.to(torch.int32), sp_edge_e.to(torch.int32), batch_time, free_time, insert_indices


def run(filename: str):
    torch.cuda.empty_cache()
    edge_starts, edge_ends, old_vertices_hash = read_edge_txt(filename, 0)
    # 在给定真是点时old_vertices_hash可以将真实点转为重排序后的点

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


    print("truss分解完成")
    print("开始equitruss构建")

    # del graph, klist, cnt
    # t1 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated())

    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equi_truss_construction(row_ptr, columns, rows, truss_result)
    torch.cuda.empty_cache()
    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equi_truss_construction(row_ptr, columns, rows, truss_result)

    ## 插入实验  random是闭区间, torchrandint为前开后闭区间 max_v为row_ptr.size(0)-2

    # group = [1,10,100,1000,10000]
    # for num in group:
    #     start_v = torch.randint(0, row_ptr.size(0)-2, (num,), device=device, dtype=torch.int32)
    #     end_v = torch.tensor([random.randint(i+1, row_ptr.size(0)-2) for i in start_v], device=device, dtype=torch.int32)

    #     row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e, batch_time, free_time, insert_indices = eqitruss_insert_batch(start_v, end_v, row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e)
    #     torch.cuda.empty_cache()
    #     print("x=x=x=x=插入删除分界线x=x=x=x=x")
    #     row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e, batch_time, free_time = eqitruss_delete_batch(insert_indices, row_ptr, columns, rows, truss_result, valid_edge, sp_node, sp_edge_s, sp_edge_e)
    #     print("-c-c-c-c-c下一组别分界线-c-c-c-c-c-")

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
    
