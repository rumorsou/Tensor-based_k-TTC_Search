import random
import time

import torch

from TETree import compress, equi_tree_construction
from utils import get_all_nbr, device
from CSRGraph import edgelist_to_CSR_gpu2, edgelist_and_truss_to_csr_gpu, read_edge_txt_gpu2, read_edge_and_truss_txt_gpu


def query_vertex_3(v: int, k: int, row_ptr, columns, max_node_id, sp_node, sp_edge_s, sp_edge_e, sp_node_id,
                   sp_node_truss, sorted_pi, idx, sp_ptr):
    # 修改后的root，各节点先指向自己，由des指向src，再compress
    # 保证合法的查询
    if k < 3:
        print("k<3")
        return
    if v < 0 or v > row_ptr.shape[0] - 1:
        print("Illegal query vertex v")
        return
    mask = torch.zeros(columns.shape[0], device=device, dtype=torch.bool)
    mask[row_ptr[v]:row_ptr[v + 1]] = True
    mask = mask | (columns == v)
    if torch.sum(mask.to(torch.int32)) == 0:
        print("No such communities")
        return

    # 初始化变量
    visited = torch.full(size=(columns.shape[0],), fill_value=-1, device=device, dtype=torch.int32)
    # max_node_id = super_graph.max_node_id
    src_edge = sp_edge_s.clone()
    des_edge = sp_edge_e.clone()
    pi = sp_node
    root = torch.full(size=(max_node_id + 1,), fill_value=-1, device=device, dtype=torch.int32)
    count = torch.full(size=(max_node_id + 2,), fill_value=0, device=device, dtype=torch.int32)

    # 更新索引，删除src/des_edge中truss<k的超节点及对应的超边
    mask1 = sp_node_truss >= k
    k_sp_node = sp_node_id[mask1]
    mask1 = src_edge >= k_sp_node[0]
    src_edge = src_edge[mask1]
    des_edge = des_edge[mask1]
    # 初始化root，使其指向自己,再让des_edge的root指向src_edge
    root[sp_node_id] = sp_node_id
    root[des_edge] = src_edge
    # compress
    compress(sp_node_id, root)
    # root[sp_node_id] = -1
    # 对于查询节点v，找到v属于的且truss>k的边

    # 找到vq超节点的根节点
    sp_node_root = torch.unique(root[pi[mask]])
    sp_node_root = sp_node_root[sp_node_truss[sp_node_root] >= k]
    # print(sp_node_root)
    if sp_node_root.size(0) == 0:
        print("No such communities")
        return

    sorted_root, idxs = torch.sort(root)
    # sp_id = sp_node_id.clone()
    # sp_id =sp_id[idxs]
    _, cnt = torch.unique_consecutive(sorted_root, return_counts=True)
    count[_] = cnt.to(torch.int32)
    count = count[:-1]
    ptr = torch.cumsum(torch.cat((torch.zeros(1, device=device, dtype=torch.int32), count), dim=0), dim=0).to(
        torch.int32)


    mask = torch.isin(sp_edge_s, sp_node_root)
    # print(sp_edge_e[mask])
    # print()
    # print(ptr.size(0))
    start = sp_node_root
    end = sp_node_root + 1
    mask = get_all_nbr(ptr[start], ptr[end])
    all_sp_node = sp_node_id[idxs][mask]
    # print(torch.sort(all_sp_node))

    start = all_sp_node
    end = all_sp_node + 1
    mask = get_all_nbr(sp_ptr[start], sp_ptr[end])
    visited[idx[mask]] = root[sorted_pi[mask]]

    return visited

def run_with_truss(filename: str, query_count: int = 1000):
    torch.cuda.empty_cache()
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

    valid_edge, sp_node, sp_edge_s, sp_edge_e, max_node_id = equi_tree_construction(row_ptr, columns, rows, truss_result)

    sorted_pi, idx = torch.sort(sp_node[valid_edge])
    sorted_truss = truss_result[valid_edge][idx]
    sp_node_id, cnt = torch.unique(sorted_pi,return_counts=True)
    max_node_id = sp_node_id[-1]
    sp_ptr = torch.cumsum(torch.cat((torch.tensor([0],device=device,dtype=cnt.dtype),cnt),dim=0),dim=0).to(torch.int32)
    sp_node_truss = sorted_truss[sp_ptr[:-1]]

    random_list = random.sample(range(0, row_ptr.size(0)-1), query_count)
    t1 = time.time()
    i=1
    for v in random_list:
        print("第"+str(i)+"次查询")
        query_vertex_3(v, 4, row_ptr ,columns, max_node_id ,sp_node ,sp_edge_s, sp_edge_e ,sp_node_id.to(torch.int32), sp_node_truss,sorted_pi,idx,sp_ptr)
        i=i+1
    t2 = time.time()
    print("Total time：",str(t2-t1),"ms")
    print("Average time：", str((t2 - t1)/query_count), "ms")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run EquiTree construction with Graph file')
    parser.add_argument('--filename', '-f',
                        default=r"facebook.txt",
                        help='Path to the truss result file')
    parser.add_argument('--query', '-q', required=True, type=int,
                        help='Number of queries to run (required)')

    args = parser.parse_args()
    run_with_truss(args.filename, name=args.name, query_count=args.query)


