import torch
from torch_scatter import segment_csr
from utils import get_all_nbr, get_all_nbr_size
import singlegpu_truss
from trusstensor import segment_add




def insert_edge(v, u, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, edge_id: torch.Tensor, truss_result: torch.Tensor):

    if v >= row_ptr.size(0):
        return False, -1, row_ptr, columns, rows, edge_id, truss_result
    
    v_nbr =  columns[row_ptr[v]:  row_ptr[v + 1]]
    mask = v_nbr-u
    if (mask == 0).sum().item() != 0:  # u已经是v的邻居了
        return False, -1
    if (mask > 0).sum().item() != 0:
        index = int(row_ptr[v]+torch.where(mask>0)[0][0])
    else:
        index = int(row_ptr[v+1])
    print("index===", index)
    # index = int(row_ptr[v + 1])
    new_tensor = torch.zeros(columns.size(0) + 1, device= columns.device, dtype= columns.dtype)
    new_tensor[:index] =  columns[:index]
    new_tensor[index + 1:] =  columns[index:]
    # 在指定位置插入新元素
    new_tensor[index] = u
    columns = new_tensor.clone()

    new_tensor[:index] =  rows[:index]
    new_tensor[index + 1:] =  rows[index:]
    # 在指定位置插入新元素
    new_tensor[index] = v
    rows = new_tensor.clone()

    new_tensor[:index] =  truss_result[:index]
    new_tensor[index + 1:] =  truss_result[index:]
    # 在指定位置插入新元素
    new_tensor[index] = 0
    truss_result = new_tensor.clone()

    row_ptr[v + 1:] += 1
    edge_id = torch.cat((edge_id, torch.tensor([edge_id.size(0)], device= edge_id.device)))
    return True, index, row_ptr, columns, rows, edge_id, truss_result





def insert_edge2(v, u, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor):
    if v >= row_ptr.size(0):
        return False, row_ptr, columns, rows
    v_nbr =  columns[row_ptr[v]:  row_ptr[v + 1]]
    print(v_nbr)
    rows1 = torch.cat((rows, torch.tensor([v], device=rows.device, dtype=torch.int32)))
    columns1 = torch.cat((columns, torch.tensor([u], device=rows.device, dtype=torch.int32)))
    columns1, ind = torch.sort(columns1)
    rows1 = rows1[ind]
    rows1, ind = torch.sort(rows1)
    columns1 = columns1[ind]
    print(rows1)
    vertices, cnt = torch.unique_consecutive(rows1, return_counts = True)
    row_ptr1 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(cnt, dim=0).to(torch.int32)))
    print(row_ptr1)
    return True, row_ptr1, columns1, rows1



def insert_equitruss(ins_edge_id, valid_edge: torch.Tensor, sp_node: torch.Tensor):
    mask = valid_edge >= ins_edge_id
    valid_edge[mask] += 1
    new_tensor = torch.zeros( sp_node.size(0) + 1, device= sp_node.device, dtype= sp_node.dtype)
    new_tensor[:ins_edge_id] =  sp_node[:ins_edge_id]
    new_tensor[ins_edge_id + 1:] =  sp_node[ins_edge_id:]
    new_tensor[ins_edge_id] = -1

    sp_node = new_tensor
    return valid_edge, sp_node


def get_affected_edge(v, u, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    device = row_ptr.device
    # 入边
    mask_v = torch.zeros( row_ptr.shape[0], dtype=torch.bool, device=device)
    mask_v[v] = True
    v_nbr_mask = mask_v[columns]  # 查找点的入边,-1一定为false，不会被选择
    # v_nbr_2 = torch.bucketize(torch.where(v_nbr_mask)[0],  row_ptr, right=True) - 1
    mask_v[v] = False
    mask_v[u] = True
    u_nbr_mask = mask_v[columns]
    v_edge_out = torch.arange(row_ptr[v],  row_ptr[v + 1], device=device)
    v_edge_in = torch.where(v_nbr_mask)[0]
    v_edges = torch.cat((v_edge_out, v_edge_in))
    
    u_edge_out = torch.arange( row_ptr[u],  row_ptr[u + 1], device=device)
    u_edge_in = torch.where(u_nbr_mask)[0]
    
    u_edges = torch.cat((u_edge_out, u_edge_in))

    v_nbr = torch.cat((columns[v_edge_out], rows[v_edge_in]))
    u_nbr = torch.cat((columns[u_edge_out], rows[u_edge_in]))
    # print(v_nbr)
    # print(u_nbr)
    w_mask1 = torch.isin(v_nbr, u_nbr)
    w_mask2 = torch.isin(u_nbr, v_nbr[w_mask1])
    # 插入边的truss值上限
    up_bound = torch.sum(w_mask1.type(torch.int32)) + 2
    # print(up_bound)
    # 找到形成三角形的邻居边
    aff_edge = torch.cat((v_edges[w_mask1], u_edges[w_mask2]))
    print(aff_edge)
    print(truss_result[aff_edge])
    print("upbound===", up_bound)
    # 筛选出邻居边中truss值小于上界的边
    mask = (truss_result[aff_edge] < up_bound)
    aff_edge = aff_edge[mask]
    print(aff_edge)

    return aff_edge

def get_affected_node_delete(del_node: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, node_truss: torch.Tensor):
    device = row_ptr.device

    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)

    rows2 = torch.cat((rows, columns))
    columns2 = torch.cat((columns, rows))
    edge_id2 = torch.cat((edge_id, edge_id))
    columns2, ind = torch.sort(columns2)
    rows2 = rows2[ind]
    edge_id2 = edge_id2[ind]
    rows2, ind = torch.sort(rows2, stable=True)
    columns2 = columns2[ind]
    edge_id2 = edge_id2[ind]
    count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows2, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr2 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32))) 
    
    node_nbr_indices, nbr_size,_=get_all_nbr_size(row_ptr2[del_node], row_ptr2[del_node+1])
    node_nbr = columns2[node_nbr_indices]
    node_src = torch.repeat_interleave(del_node, nbr_size)
    node_nbr = node_nbr[node_truss[node_nbr]<node_truss[node_src]]
    node_nbr = node_nbr[node_nbr!=-1]

    return node_nbr


def get_affected_edge2(insert_mask: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    device = row_ptr.device

    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    insert_edges = edge_id[insert_mask]

    h_edge, mask_insert, h_row_ptr, h_columns, h_rows = get_sub_graph2(insert_edges, row_ptr, columns, rows, edge_id)
    h_support = singlegpu_truss.support_computing(h_rows, h_columns, h_row_ptr, 1)
    sub_support = h_support[mask_insert]+2
    # print(insert_edges)
    # print("上界===",sub_support)

    rows2 = torch.cat((rows, columns))
    columns2 = torch.cat((columns, rows))
    edge_id2 = torch.cat((edge_id, edge_id))
    columns2, ind = torch.sort(columns2)
    rows2 = rows2[ind]
    edge_id2 = edge_id2[ind]
    rows2, ind = torch.sort(rows2, stable=True)
    columns2 = columns2[ind]
    edge_id2 = edge_id2[ind]
    count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows2, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr2 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32))) 
    
    v = rows[insert_mask]
    u = columns[insert_mask]
    v_nbr_indices, v_size,_=get_all_nbr_size(row_ptr2[v], row_ptr2[v+1])
    u_nbr_indices, u_size,_=get_all_nbr_size(row_ptr2[u], row_ptr2[u+1])
    v_insert_edge = torch.repeat_interleave(insert_edges, v_size)
    u_insert_edge = torch.repeat_interleave(insert_edges, u_size)
    off = 1000000000
    v_nbr = columns2[v_nbr_indices].to(torch.long)+v_insert_edge.to(torch.long)*off
    u_nbr = columns2[u_nbr_indices].to(torch.long)+u_insert_edge.to(torch.long)*off
    w_mask1 = torch.isin(v_nbr, u_nbr)
    w_mask2 = torch.isin(u_nbr, v_nbr[w_mask1])
    v_aff_edge = edge_id2[v_nbr_indices][w_mask1]
    u_aff_edge = edge_id2[u_nbr_indices][w_mask2]

    truss_result2 = truss_result.clone()
    truss_result2[insert_edges] = sub_support
    v_aff_edge = v_aff_edge[truss_result2[v_aff_edge]<truss_result2[v_insert_edge[w_mask1]]]
    u_aff_edge = u_aff_edge[truss_result2[u_aff_edge]<truss_result2[u_insert_edge[w_mask2]]]

    aff_edge = torch.unique(torch.cat((v_aff_edge, u_aff_edge)))

    return aff_edge


def get_affected_edge3(insert_mask: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    device = row_ptr.device

    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    insert_edges = edge_id[insert_mask]

    rows2 = torch.cat((rows, columns))
    columns2 = torch.cat((columns, rows))
    edge_id2 = torch.cat((edge_id, edge_id))
    columns2, ind = torch.sort(columns2)
    rows2 = rows2[ind]
    edge_id2 = edge_id2[ind]
    rows2, ind = torch.sort(rows2, stable=True)
    columns2 = columns2[ind]
    edge_id2 = edge_id2[ind]
    count = torch.zeros(row_ptr.size(0)-1, device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows2, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr2 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32))) 
    
    v = rows[insert_mask]
    u = columns[insert_mask]
    v_nbr_indices, v_size, v_ptr=get_all_nbr_size(row_ptr2[v], row_ptr2[v+1])
    u_nbr_indices, u_size, u_ptr=get_all_nbr_size(row_ptr2[u], row_ptr2[u+1])
    v_insert_edge = torch.repeat_interleave(insert_edges, v_size)
    u_insert_edge = torch.repeat_interleave(insert_edges, u_size)
    off = 1000000000
    v_nbr = columns2[v_nbr_indices].to(torch.long)+v_insert_edge.to(torch.long)*off
    u_nbr = columns2[u_nbr_indices].to(torch.long)+u_insert_edge.to(torch.long)*off
    w_mask1 = torch.isin(v_nbr, u_nbr)
    w_mask2 = torch.isin(u_nbr, v_nbr[w_mask1])
    v_aff_edge = edge_id2[v_nbr_indices][w_mask1]
    u_aff_edge = edge_id2[u_nbr_indices][w_mask2]
    
    tri_size = segment_csr(w_mask1.to(torch.long), v_ptr.to(torch.long))
    tri_ptr = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(tri_size, dim=0).to(torch.int32))) 
    v_aff_truss = truss_result[v_aff_edge]
    u_aff_truss = truss_result[u_aff_edge]
    print(v_aff_truss)
    print(u_aff_truss)
    k_tri_truss = torch.where(v_aff_truss<u_aff_truss, v_aff_truss, u_aff_truss)
    mask = k_tri_truss!=-1
    k_tri_truss = k_tri_truss[mask]
    tri_size = segment_csr(mask.to(torch.long), tri_ptr.to(torch.long))

    tri_insert_edge = torch.repeat_interleave(insert_edges, tri_size)
    k_tri_truss = k_tri_truss.to(torch.long)+tri_insert_edge.to(torch.long)*off
    unique_tri_truss, k_tri_count=torch.unique(k_tri_truss, return_counts = True)
    tri_insert_edge = unique_tri_truss/off
    tri_insert_edge = tri_insert_edge.to(torch.int32)
    print("插入边==", tri_insert_edge)
    unique_tri_truss = unique_tri_truss%off
    print("三角形k值==",unique_tri_truss)
    
    # print(k_tri_count)
    count = torch.zeros(edge_id.size(0), device=device, dtype=torch.int32)
    tri_insert_unique, cnt = torch.unique_consecutive(tri_insert_edge, return_counts = True)
    count[tri_insert_unique] = cnt.to(torch.int32)
    count = count[insert_edges]
    tri_ptr = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32)))
    tri_ptr2 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count.flip(0), dim=0).to(torch.int32)))
    k_tri_cumsum = torch.zeros(k_tri_count.size(0), device=device, dtype=torch.int32)
    k_tri_count2 = k_tri_count.flip(0)
    # print(tri_ptr2)
    for i in range(tri_ptr2.size(0)-1):
        k_tri_cumsum[tri_ptr2[i]:tri_ptr2[i+1]] = torch.cumsum(k_tri_count2[tri_ptr2[i]:tri_ptr2[i+1]],dim=0).to(torch.int32)
    # print(k_tri_cumsum)
    # print(tri_ptr)

    ###  上界中unique_tri_truss需要转为连续的值，保证上界的正确性，即插入边的上界不能只出现于unique_tri_truss中
    # segment_add(k_tri_count.flip(0).to(torch.int32), tri_ptr2, k_tri_cumsum)
    # print(k_tri_cumsum)
    k_tri_cumsum+=2 # 按照trussness计算
    k_tri_cumsum = k_tri_cumsum.flip(0)  # 反转
    print("k三角形数量==",k_tri_cumsum)
    mask = k_tri_cumsum>=(unique_tri_truss-1)
    k_tri_cumsum[~mask] = 0
    upper_bound= segment_csr(k_tri_cumsum.to(torch.long), tri_ptr.to(torch.long), reduce="max")
    # cumsum可能反了
    print("上界===",upper_bound)
    truss_result2 = truss_result.clone()
    truss_result2[insert_edges] = upper_bound.to(torch.int32)  # 这个sub_support作为上界
    v_aff_edge = v_aff_edge[truss_result2[v_aff_edge]<truss_result2[v_insert_edge[w_mask1]]]
    u_aff_edge = u_aff_edge[truss_result2[u_aff_edge]<truss_result2[u_insert_edge[w_mask2]]]

    aff_edge = torch.unique(torch.cat((v_aff_edge, u_aff_edge)))

    return aff_edge

def get_affected_edge4(insert_mask: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
    device = row_ptr.device

    edge_id = torch.arange(0, columns.size(0), device=device, dtype=torch.int32)
    insert_edges = edge_id[insert_mask]

    # h_edge, mask_insert, h_row_ptr, h_columns, h_rows = get_sub_graph2(insert_edges, row_ptr, columns, rows, edge_id)
    # h_support = singlegpu_truss.support_computing(h_rows, h_columns, h_row_ptr, 1)
    # sub_support = h_support[mask_insert]+2
    # print(insert_edges)
    # print(sub_support)

    rows2 = torch.cat((rows, columns))
    columns2 = torch.cat((columns, rows))
    edge_id2 = torch.cat((edge_id, edge_id))
    columns2, ind = torch.sort(columns2)
    rows2 = rows2[ind]
    edge_id2 = edge_id2[ind]
    rows2, ind = torch.sort(rows2, stable=True)
    columns2 = columns2[ind]
    edge_id2 = edge_id2[ind]
    count = torch.zeros(row_ptr.size(0), device=device, dtype=torch.int32)
    vertices, cnt = torch.unique_consecutive(rows2, return_counts = True)
    count[vertices] = cnt.to(torch.int32)
    row_ptr2 = torch.cat((torch.tensor([0], device=rows.device, dtype=torch.int32), torch.cumsum(count, dim=0).to(torch.int32))) 
    
    v = rows[insert_mask]
    u = columns[insert_mask]
    v_nbr_indices, v_size=get_all_nbr_size(row_ptr2[v], row_ptr2[v+1])
    u_nbr_indices, u_size=get_all_nbr_size(row_ptr2[u], row_ptr2[u+1])
    v_insert_edge = torch.repeat_interleave(insert_edges, v_size)
    u_insert_edge = torch.repeat_interleave(insert_edges, u_size)
    off = 1000000000
    v_nbr = columns2[v_nbr_indices].to(torch.long)+v_insert_edge.to(torch.long)*off
    u_nbr = columns2[u_nbr_indices].to(torch.long)+u_insert_edge.to(torch.long)*off
    w_mask1 = torch.isin(v_nbr, u_nbr)
    w_mask2 = torch.isin(u_nbr, v_nbr[w_mask1])
    v_aff_edge = edge_id2[v_nbr_indices][w_mask1]
    u_aff_edge = edge_id2[u_nbr_indices][w_mask2]

    # truss_result2 = truss_result.clone()
    # truss_result2[insert_edges] = sub_support
    # v_aff_edge = v_aff_edge[truss_result2[v_aff_edge]<truss_result2[v_insert_edge[w_mask1]]]
    # u_aff_edge = u_aff_edge[truss_result2[u_aff_edge]<truss_result2[u_insert_edge[w_mask2]]]

    aff_edge = torch.unique(torch.cat((v_aff_edge, u_aff_edge)))

    return aff_edge

# def get_affected_edge3(insert_mask: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, truss_result: torch.Tensor):
#     device = row_ptr.device

#     v = rows[insert_mask]
#     u = columns[insert_mask]
#     mask_v = torch.zeros(row_ptr.shape[0], dtype=torch.bool, device=device)
#     mask_v[v] = True
#     v_nbr_mask = mask_v[columns]  # 查找点的入边,-1一定为false，不会被选择
#     # v_nbr_2 = torch.bucketize(torch.where(v_nbr_mask)[0],  row_ptr, right=True) - 1
#     mask_v[v] = False
#     mask_v[u] = True
#     u_nbr_mask = mask_v[columns]
#     v_edge_out = get_all_nbr(row_ptr[v], row_ptr[v+1])
#     v_edge_in = torch.where(v_nbr_mask)[0]
#     v_edges = torch.cat((v_edge_out, v_edge_in))
    
#     u_edge_out = get_all_nbr(row_ptr[u], row_ptr[u+1])
#     u_edge_in = torch.where(u_nbr_mask)[0]
    
#     u_edges = torch.cat((u_edge_out, u_edge_in))

#     v_nbr = torch.cat((columns[v_edge_out], rows[v_edge_in]))
#     u_nbr = torch.cat((columns[u_edge_out], rows[u_edge_in]))
#     # print(v_nbr)
#     # print(u_nbr)
#     w_mask1 = torch.isin(v_nbr, u_nbr)
#     w_mask2 = torch.isin(u_nbr, v_nbr[w_mask1])
#     # 插入边的truss值上限
#     # up_bound = torch.sum(w_mask1.type(torch.int32)) + 2
#     # print(up_bound)
#     # 找到形成三角形的邻居边
#     aff_edge = torch.cat((v_edges[w_mask1], u_edges[w_mask2]))
#     # print(aff_edge)
#     # print(truss_result[aff_edge])
#     # print("upbound===", up_bound)
#     # 筛选出邻居边中truss值小于上界的边
#     # mask = (truss_result[aff_edge] < up_bound)
#     # aff_edge = aff_edge[mask]
#     # print(aff_edge)

#     return aff_edge

def get_sub_graph(aff_edge: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, edge_id:torch.Tensor):
    device = row_ptr.device
    start_v = rows[aff_edge]
    # 获取被删除的边的终止点id
    end_v =  columns[aff_edge]
    p = torch.unique(torch.cat([start_v, end_v]))  # p即为所有点
    mask_v = torch.zeros( row_ptr.shape[0], dtype=torch.bool, device=device)
    mask_v[p] = True
    mask = mask_v[ columns]  # 查找点的入边,-1一定为false，不会被选择
    sub_edge_out = get_all_nbr(row_ptr[p], row_ptr[p + 1])  # 出边索引
    mask[sub_edge_out] = True

    sub_edge =  edge_id[mask]  # 保证subedge是从小到大的
    sub_columns =  columns[mask]
    sub_rows = rows[mask]
    sub_row_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.int64),
                             torch.cumsum(segment_csr(mask.int(), row_ptr.to(torch.int64)), dim=0)))
    border_edge_mask = torch.zeros(edge_id.shape[0], dtype=torch.bool, device=device)
    border_edge_mask[sub_edge] = True
    border_edge_mask[aff_edge] = False
    border_edge =  edge_id[border_edge_mask]
    return sub_edge, border_edge, sub_row_ptr.to(torch.int32), sub_columns, sub_rows


def get_sub_graph2(aff_edge: torch.Tensor, row_ptr: torch.Tensor, columns: torch.Tensor, rows: torch.Tensor, edge_id:torch.Tensor):
    device = row_ptr.device
    start_v = rows[aff_edge]
    # 获取被删除的边的终止点id
    end_v =  columns[aff_edge]
    p = torch.unique(torch.cat([start_v, end_v]))  # p即为所有点
    mask_v = torch.zeros( row_ptr.shape[0], dtype=torch.bool, device=device)
    mask_v[p] = True
    mask = mask_v[columns]  # 查找点的入边,-1一定为false，不会被选择
    sub_edge_out = get_all_nbr(row_ptr[p], row_ptr[p + 1])  # 出边索引
    mask[sub_edge_out] = True

    sub_edge =  edge_id[mask]  # 保证subedge是从小到大的
    sub_columns =  columns[mask]
    sub_rows = rows[mask]
    sub_row_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.int64),
                             torch.cumsum(segment_csr(mask.int(), row_ptr.to(torch.int64)), dim=0)))
    mask2 = torch.zeros(edge_id.shape[0], dtype=torch.bool, device=device)
    mask2[aff_edge] = True
    mask_aff = mask2[sub_edge]

    return sub_edge, mask_aff, sub_row_ptr.to(torch.int32), sub_columns, sub_rows