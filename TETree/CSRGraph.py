import time

import numpy as np


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time} 秒")
        return result

    return wrapper


class max_vertex:
    value = 0


'''
naive method to read edge and transfer csr in cpu
'''
@calculate_time
def read_edge_txt(filename="graph/example-graph1.txt", dataset_type=0):
    # 读取txt或csv格式的数据，分隔符默认是空格
    global index

    time1 = time.time()
    array = np.loadtxt(filename, dtype=np.int32)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]
    time2 = time.time()
    print("读文件时间:", time2 - time1, "!!!")

    # x = np.concatenate((edge_starts, edge_ends))
    # x, cnt = np.unique(x, return_counts=True)
    # print(np.max(cnt))

    print("边数："+str(len(edge_starts)))
    print("finish read!")

    # 去除自循环边, 可以消除部分点
    if dataset_type == 0:  # 有向图
        mask = edge_starts != edge_ends
    else:
        mask = edge_starts < edge_ends
    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]

    # 统计点数
    all_edges = np.concatenate((edge_starts, edge_ends), axis=0)

    vertices, counts = np.unique(all_edges, return_counts=True)

    # 去除两点孤立的孤立边， 只去除一次
    leaf_mask = counts == 1  # 找到只出现一次的点
    leaf = vertices[leaf_mask]  # 获取孤立点
    if leaf.size > 0:
        mask1 = np.isin(edge_starts, leaf, invert=True)  # 获取非孤立点
        mask2 = np.isin(edge_ends, leaf, invert=True)

        mask = mask1 & mask2
        edge_starts = edge_starts[mask]
        edge_ends = edge_ends[mask]

        vertices = np.unique(np.concatenate((edge_starts, edge_ends)))

    vertex_to_index = {}
    # 将点映射为从0开始的连续的id
    for vertex, index in zip(vertices, range(len(vertices))):
        vertex_to_index[vertex] = index

    max_vertex.value = index

    # 修改edgestart和edgeend，主要是就是编号
    for index in range(len(edge_starts)):
        start = vertex_to_index[edge_starts[index]]
        end = vertex_to_index[edge_ends[index]]
        edge_starts[index] = start
        edge_ends[index] = end

    # print(edge_starts)
    # print(edge_ends)
    return edge_starts, edge_ends, vertex_to_index


# edge_starts: torch.Tensor, edge_ends: torch.Tensor
@calculate_time
def edgelist_to_CSR(edge_starts: np.ndarray, edge_ends: np.ndarray, direct=False):
    """
    :param direct:
    :param edge_starts: [ 1,  1,  1,  1,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  6,  7, 7,  7,  7,  8,  8,  8,  9,  9, 10]
    :param edge_ends:   [ 2,  3,  4,  5,  3,  4,  4,  7,  9,  5,  6,  7,  6,  7,  7,  8, 11,  8, 9, 10, 11,  9, 10, 11, 10, 11, 11]
    :param vertices: [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    """

    # print("--------edgelist_to_CSR--------")
    if direct:

        # 变为由小序号指向大序号的有向图
        temp_indices = edge_starts > edge_ends
        temp_store = edge_starts[temp_indices]
        edge_starts[temp_indices] = edge_ends[temp_indices]
        edge_ends[temp_indices] = temp_store

        # 排序
        indices_mask = np.argsort(edge_ends)

        # 现按照columns排序
        columns = edge_ends[indices_mask]
        source_vertices = edge_starts[indices_mask]

        indices2 = np.argsort(source_vertices, kind='stable')
        source_vertices = source_vertices[indices2]
        columns = columns[indices2]
        

        time1 = time.time()
        # 获取连续的唯一值以及数量
        unique_vertices_id, breakpoints = np.unique(source_vertices, return_counts=True)


        full_vertices_id = np.zeros((max_vertex.value + 1,), dtype=int)
        full_vertices_id[unique_vertices_id] = breakpoints

        row_ptr = np.concatenate((np.array([0]), np.cumsum(full_vertices_id, axis=0)))
        time2 = time.time()
        print("csr构建时间:", time2 - time1,"!!!")

        return row_ptr, columns, source_vertices

    else:
        source_vertices = np.concatenate((edge_starts, edge_ends), axis=0)
        destination_vertices = np.concatenate((edge_ends, edge_starts), axis=0)
        column_to_edge = np.concatenate((np.arange(0, edge_starts.size),
                                         np.arange(0, edge_starts.size)))

        # 排序
        indices_mask = np.argsort(source_vertices)

        # 按索引对应位置
        source_vertices = source_vertices[indices_mask]
        columns = destination_vertices[indices_mask]
        column_to_edge = column_to_edge[indices_mask]

        # 获取连续的唯一值以及数量
        unique_vertices_id, breakpoints = np.unique(source_vertices, return_counts=True)

        # m个点
        # 计算输入 Tensor 的前缀和，长度为m
        breakpoints = np.cumsum(breakpoints, axis=0)

        # 完全排序
        row_ptr = np.concatenate((np.array([0]), breakpoints))

        for index in range(1, row_ptr.size):
            start = row_ptr[index - 1]
            end = row_ptr[index]

            indices = np.argsort(columns[start: end])
            temp = columns[start: end][indices]
            columns[start: end] = temp
            column_to_edge[start: end] = column_to_edge[start:end][indices]

        return row_ptr, columns, column_to_edge
    



'''
new method to read edge and transfer csr in gpu
'''

import torch
import time

@calculate_time
def read_edge_txt_gpu2(filename="graph/example-graph1.txt", dataset_type=0):
    global index

    time1 = time.time()
    array = np.loadtxt(filename, dtype=np.int32)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]
    time2 = time.time()
    print("读文件时间:", time2 - time1, "!!!")

    return edge_starts, edge_ends   # 返回的为np格式


@calculate_time
def read_edge_and_truss_txt_gpu(filename="graph/example-graph1.txt", dataset_type=0):
    global index

    time1 = time.time()
    array = np.loadtxt(filename, delimiter=',', dtype=np.int32)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]
    truss = array[:, 2]
    time2 = time.time()
    print("读文件时间:", time2 - time1, "!!!")

    return edge_starts, edge_ends, truss   # 返回的为np格式



@calculate_time
def edgelist_to_CSR_gpu2(edge_starts: np.ndarray, edge_ends: np.ndarray, direct=False):
    # 将数据移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"已使用显存: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    edge_starts = torch.from_numpy(edge_starts).to(device)
    edge_ends = torch.from_numpy(edge_ends).to(device)

    mask = edge_starts != edge_ends
    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]     # 3x
    del mask

    # 使用torch.unique替代np.unique
    # torch.unique得到的unique，本身就是排过序的
    vertices, inverse_indices, counts = torch.unique(torch.cat((edge_starts, edge_ends), dim=0),return_inverse=True, return_counts=True) # 8x

    # 去除两点孤立的孤立边， 只去除一次
    print(f"已使用显存: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")


    shapex = edge_starts.shape[0]
    del edge_starts, edge_ends
    edge_starts = inverse_indices[:shapex]  # 在这里新边顺序还是一致的，从下边开始，会修改边的顺序
    edge_ends = inverse_indices[shapex:]
    del inverse_indices, counts

    # 变为由小序号指向大序号的有向图
    temp_indices = edge_starts > edge_ends
    temp_store = edge_starts[temp_indices]
    edge_starts[temp_indices] = edge_ends[temp_indices]
    edge_ends[temp_indices] = temp_store
    del temp_indices, temp_store

    time1 = time.time() 
    columns, indices_mask = torch.sort(edge_ends)
    rows = edge_starts[indices_mask]
    rows, indices2 = torch.sort(rows, stable=True)
    columns = columns[indices2]
    del indices_mask, indices2
    
    # 获取连续的唯一值以及数量
    unique_vertices_id, breakpoints = torch.unique_consecutive(
        rows, return_counts=True)

    # 即，rowptr中包含所有的点，但是后续一排的大编号的点没有出边
    max_vertex = torch.max(torch.cat([edge_starts, edge_ends])).item()
    full_vertices_id = torch.zeros(max_vertex + 1, dtype=torch.int32, device=device) # [0,maxvertex+1)
    full_vertices_id[unique_vertices_id] = breakpoints.to(torch.int32)
    # 计算累积和
    row_ptr = torch.cat([torch.tensor([0], device=device), 
                       torch.cumsum(full_vertices_id, dim=0)])
    
    time2 = time.time()
    print("gpu上csr构建时间:", time2 - time1,"!!!")

    print(row_ptr.dtype, columns.dtype, rows.dtype)
    return row_ptr.to(torch.int32), columns.to(torch.int32), rows.to(torch.int32)


@calculate_time
def edgelist_and_truss_to_csr_gpu(edge_starts: np.ndarray, edge_ends: np.ndarray, truss: np.ndarray, direct=False):
    # 将数据移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_starts = torch.from_numpy(edge_starts).to(device)
    edge_ends = torch.from_numpy(edge_ends).to(device)
    truss = torch.from_numpy(truss).to(device)

    mask = edge_starts != edge_ends
    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]
    truss = truss[mask]

    all_edges = torch.cat((edge_starts, edge_ends), dim=0)
    # 使用torch.unique替代np.unique
    # torch.unique得到的unique，本身就是排过序的
    vertices, inverse_indices, counts = torch.unique(all_edges,return_inverse=True, return_counts=True)


    shapex = edge_starts.shape[0]
    del edge_starts, edge_ends
    edge_starts = inverse_indices[:shapex]  # 在这里新边顺序还是一致的，从下边开始，会修改边的顺序
    edge_ends = inverse_indices[shapex:]
    del inverse_indices, counts

    # 变为由小序号指向大序号的有向图
    temp_indices = edge_starts > edge_ends
    temp_store = edge_starts[temp_indices]
    edge_starts[temp_indices] = edge_ends[temp_indices]
    edge_ends[temp_indices] = temp_store
    del temp_indices, temp_store

    time1 = time.time() 
    columns, indices_mask = torch.sort(edge_ends)
    rows = edge_starts[indices_mask]
    truss = truss[indices_mask]
    rows, indices2 = torch.sort(rows, stable=True)
    columns = columns[indices2]
    truss = truss[indices2]
    del indices_mask, indices2
    
    # 获取连续的唯一值以及数量
    unique_vertices_id, breakpoints = torch.unique_consecutive(
        rows, return_counts=True)

    # 即，rowptr中包含所有的点，但是后续一排的大编号的点没有出边
    max_vertex = torch.max(torch.cat([edge_starts, edge_ends])).item()
    full_vertices_id = torch.zeros(max_vertex + 1, dtype=torch.int32, device=device)
    full_vertices_id[unique_vertices_id] = breakpoints.to(torch.int32)
    # 计算累积和
    row_ptr = torch.cat([torch.tensor([0], device=device), 
                       torch.cumsum(full_vertices_id, dim=0)])
    
    time2 = time.time()
    print("gpu上csr构建时间:", time2 - time1,"!!!")

    # print(row_ptr.dtype, columns.dtype, rows.dtype)
    return row_ptr.to(torch.int32), columns.to(torch.int32), rows.to(torch.int32), truss.to(torch.int32)

@calculate_time
def edgelist_to_CSR_gpu2_map(edge_starts: np.ndarray, edge_ends: np.ndarray, direct=False):
    # 将数据移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_starts = torch.from_numpy(edge_starts).to(device)
    edge_ends = torch.from_numpy(edge_ends).to(device)

    mask = edge_starts != edge_ends
    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]
    del mask

    all_edges = torch.cat((edge_starts, edge_ends), dim=0)
    # 使用torch.unique替代np.unique
    # torch.unique得到的unique，本身就是排过序的
    vertices, inverse_indices, counts = torch.unique(all_edges,return_inverse=True, return_counts=True)


    shapex = edge_starts.shape[0]
    del edge_starts, edge_ends
    edge_starts = inverse_indices[:shapex].to(torch.int32)  # 在这里新边顺序还是一致的，从下边开始，会修改边的顺序
    edge_ends = inverse_indices[shapex:].to(torch.int32)
    del inverse_indices, counts


    # 变为由小序号指向大序号的有向图
    temp_indices = edge_starts > edge_ends
    temp_store = edge_starts[temp_indices]
    edge_starts[temp_indices] = edge_ends[temp_indices]
    edge_ends[temp_indices] = temp_store
    del temp_indices, temp_store

    time1 = time.time() 
    columns, indices_mask = torch.sort(edge_ends)
    rows = edge_starts[indices_mask]
    rows, indices2 = torch.sort(rows, stable=True)
    columns = columns[indices2]
    del indices_mask, indices2
    
    # 获取连续的唯一值以及数量
    unique_vertices_id, breakpoints = torch.unique_consecutive(
        rows, return_counts=True)

    # 即，rowptr中包含所有的点，但是后续一排的大编号的点没有出边
    max_vertex = torch.max(torch.cat([edge_starts, edge_ends])).item()
    full_vertices_id = torch.zeros(max_vertex + 1, dtype=torch.int32, device=device)
    full_vertices_id[unique_vertices_id] = breakpoints.to(torch.int32)
    # 计算累积和
    row_ptr = torch.cat([torch.tensor([0], device=device), 
                       torch.cumsum(full_vertices_id, dim=0)])
    
    time2 = time.time()
    print("gpu上csr构建时间:", time2 - time1,"!!!")

    edge_map = {}
    # 将GPU上的tensor转移到CPU并转换为numpy数组
    rows_cpu = rows.cpu().numpy()
    columns_cpu = columns.cpu().numpy()
    
    # 使用numpy数组进行迭代
    index_to_vertex = {idx: vertex.item() for idx, vertex in enumerate(vertices)}
    for i, (s, e) in enumerate(zip(rows_cpu, columns_cpu)):
        edge_map[(index_to_vertex[s], index_to_vertex[e])] = i
    print(row_ptr.dtype, columns.dtype, rows.dtype)
    return row_ptr.to(torch.int32), columns.to(torch.int32), rows.to(torch.int32), edge_map
    

if __name__ == "__main__":
    edge_starts, edge_ends = read_edge_txt_gpu2("/home/featurize/work/2.18/com-amazon_zero.el", dataset_type=0)
    row_ptr, columns, rows = edgelist_to_CSR_gpu2(edge_starts, edge_ends, direct=True)
