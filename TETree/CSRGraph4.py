import time

import numpy as np

# import torch

# from funcUtils import my_repeat2

# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

'''

这里的所有存储均采用int32类型

这里修改了columns的排序方式，使用基数排序确保有序
'''


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


@calculate_time
def read_edge_txt(filename="graph/example-graph1.txt", dataset_type=0):
    # 读取txt或csv格式的数据，分隔符默认是空格
    global index
    array = np.loadtxt(filename, dtype=np.int32)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]

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
    #
    # print(edge_starts)
    # print(edge_ends)
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
        

        # indices1 = np.argsort(columns)
        # columns = columns[indices1]
        # source_vertices = columns[indices1]

        # indices2 = np.argsort(source_vertices, kind='stable')

        # 获取连续的唯一值以及数量
        unique_vertices_id, breakpoints = np.unique(source_vertices, return_counts=True)

        # m个点
        # 计算输入 Tensor 的前缀和，长度为m
        # cumsum_points = np.cumsum(breakpoints, axis=0)

        # 完全排序
        # row_ptr = np.concatenate((np.array([0]), cumsum_points))

        # for index in range(1, row_ptr.size):
        #     start = row_ptr[index - 1]
        #     end = row_ptr[index]

        #     indices = np.argsort(columns[start: end])
        #     temp = columns[start: end][indices]
        #     columns[start: end] = temp

        ## 因为直接返回有向图，所以unique_vertices_id 必定不连续
        full_vertices_id = np.zeros((max_vertex.value + 1,), dtype=int)
        full_vertices_id[unique_vertices_id] = breakpoints

        row_ptr = np.concatenate((np.array([0]), np.cumsum(full_vertices_id, axis=0)))

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

        # diff = np.concatenate((np.diff(unique_vertices_id), np.array([1])))  # m
        #
        # columns = torch.tensor(columns, device=device, dtype=torch.int32)
        # column_to_edge = torch.tensor(column_to_edge, device=device, dtype=torch.int32)
        # diff = torch.tensor(diff, device=device, dtype=torch.int32)
        #
        # # row_ptr = np.concatenate((np.array([0]), np.flip(my_repeat2(np.flip(breakpoints), np.flip(diff)))))
        # row_ptr = torch.cat(
        #     (torch.tensor([0], dtype=torch.int32), torch.repeat_interleave(breakpoints.flip(0), diff.flip(0)).flip(0)))

        return row_ptr, columns, column_to_edge



