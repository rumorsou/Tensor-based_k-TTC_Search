import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

# 去重用的off偏移值
# off在这里最多为10000000
# off = 10000000


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time} 秒")
        return result

    return wrapper


def get_all_nbr(starts: torch.Tensor, nexts: torch.Tensor):
    """
    :param starts:[1,1,1,5,5]
    :param nexts: [4,4,4,8,8]
    :return:
    """
    sizes = nexts - starts
    nbr_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.int32),
                         torch.cumsum(sizes, dim=0, dtype=torch.int32)))  # [0,3,6,9,12,15]
    # starts_indices = nbr_ptr[:-1]  # [0,3,6,9,12]

    # 从索引到点
    nbr = torch.arange(int(nbr_ptr[-1]), device=device, dtype=torch.int32) - torch.repeat_interleave(
        nbr_ptr[:-1] - starts, sizes)
    return nbr

def get_all_nbr_size(starts: torch.Tensor, nexts: torch.Tensor):
    """
    :param starts:[1,1,1,5,5]
    :param nexts: [4,4,4,8,8]
    :return:
    """
    sizes = nexts - starts
    nbr_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.int32),
                         torch.cumsum(sizes, dim=0, dtype=torch.int32)))  # [0,3,6,9,12,15]
    # starts_indices = nbr_ptr[:-1]  # [0,3,6,9,12]

    # 从索引到点
    nbr = torch.arange(int(nbr_ptr[-1]), device=device, dtype=torch.int32) - torch.repeat_interleave(
        nbr_ptr[:-1] - starts, sizes)
    return nbr, sizes, nbr_ptr

def get_all_nbr_cpu(starts: torch.Tensor, nexts: torch.Tensor):
    """
    :param starts:[1,1,1,5,5]
    :param nexts: [4,4,4,8,8]
    :return:
    """
    sizes = nexts - starts
    nbr_ptr = torch.cat((torch.tensor([0], device=cpu, dtype=torch.int32),
                         torch.cumsum(sizes, dim=0, dtype=torch.int32)))  # [0,3,6,9,12,15]
    # starts_indices = nbr_ptr[:-1]  # [0,3,6,9,12]

    # 从索引到点
    nbr = torch.arange(int(nbr_ptr[-1]), device=cpu, dtype=torch.int32) - torch.repeat_interleave(
        nbr_ptr[:-1] - starts, sizes)
    return nbr

def sort_isin(tensor1: torch.Tensor,tensor2: torch.Tensor):
    sort_cat, ind = torch.sort(torch.cat((tensor2, tensor1)), stable=True)
    offset = tensor2.size(0)
    uvw_mask = torch.zeros(tensor1.size(0), device=device, dtype=torch.bool)
    kind = torch.cat((torch.ones(tensor2.size(0),device=device, dtype=torch.bool), torch.zeros(tensor1.size(0),device=device, dtype=torch.bool)))[ind]
    uniques, reverse, cnt = torch.unique_consecutive(sort_cat, return_counts=True, return_inverse=True)
    ptr = reverse[kind]
    r_ptr= torch.cumsum(torch.cat((torch.tensor([0],device=device,dtype=torch.int32),cnt),dim=0),dim=0)
    ind = ind[get_all_nbr(r_ptr[ptr], r_ptr[ptr+1])]
    ind = ind[ind>=offset]-offset
    uvw_mask[ind] = True
    return uvw_mask
    # print(uvw_mask)

# def sp_edge_unique(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     merge = tensor1 * off + tensor2
#     merge = torch.unique(merge)
#     tensor1 = (merge / off).type(torch.long)
#     tensor2 = merge - tensor1 * off
#     return tensor1, tensor2


# @calculate_time
def insert_graph_batch(tensor1: torch.Tensor, tensor2: torch.Tensor, insert_mask: torch.Tensor):
    """
    不使用off的segment_unique。
    """
    tensor2, ind = torch.sort(tensor2)
    tensor1 = tensor1[ind]
    insert_mask = insert_mask[ind]
    tensor1, ind = torch.sort(tensor1, stable=True)
    tensor2 = tensor2[ind]
    insert_mask = insert_mask[ind]
    '''
    这里有个隐藏逻辑，最终结果是按照tensor1的升序去做的
    '''
    diff2 = torch.cat((torch.ones(1, device=device, dtype=torch.int32), torch.diff(tensor2)))
    diff1 = torch.cat((torch.ones(1, device=device, dtype=torch.int32), torch.diff(tensor1)))
    mask = (diff2 != 0) | (diff1 != 0)
    return tensor1[mask], tensor2[mask], insert_mask[mask]

def sp_edge_unique_ascending(tensor1: torch.Tensor, tensor2: torch.Tensor):
    device1 = tensor1.device
    """
    不使用off的segment_unique。
    """
    tensor2, ind = torch.sort(tensor2)
    tensor1 = tensor1[ind]
    tensor1, ind = torch.sort(tensor1, stable=True)
    tensor2 = tensor2[ind]
    '''
    这里有个隐藏逻辑，最终结果是按照tensor1的升序去做的
    '''
    diff2 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor2)))
    diff1 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor1)))
    mask = (diff2 != 0) | (diff1 != 0)
    return tensor1[mask], tensor2[mask]

# @calculate_time
def sp_edge_unique_descending_v1(tensor1: torch.Tensor, tensor2: torch.Tensor):
    device1 = tensor1.device
    """
    不使用off的segment_unique。
    """
    tensor2, ind = torch.sort(tensor2, descending=True)
    tensor1 = tensor1[ind]
    tensor1, ind = torch.sort(tensor1, descending=True, stable=True)
    tensor2 = tensor2[ind]
    # tensor1 = tensor1.flip(0)
    # tensor2 = tensor2.flip(0)
    '''
    按tensor1和tensor2的降序去做
    '''
    diff2 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor2)))
    diff1 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor1)))
    mask = (diff2 != 0) | (diff1 != 0)
    return tensor1[mask], tensor2[mask]

def sp_edge_unique_descending(tensor1: torch.Tensor, tensor2: torch.Tensor):
    device1 = tensor1.device
    """
    不使用off的segment_unique。
    """
    key = tensor1.to(torch.int64) * (tensor2.max() + 1) + tensor2
    key, ind = torch.sort(key, descending=True)
    tensor1 = tensor1[ind]
    tensor2 = tensor2[ind]
    del ind
    diff = torch.cat([torch.ones(1, device=tensor1.device, dtype=torch.bool),
                      key[1:] != key[:-1]])
    return tensor1[diff], tensor2[diff]
    # tensor2, ind = torch.sort(tensor2, descending=True)
    # tensor1 = tensor1[ind]
    # tensor1, ind = torch.sort(tensor1, descending=True, stable=True)
    # tensor2 = tensor2[ind]
    # # tensor1 = tensor1.flip(0)
    # # tensor2 = tensor2.flip(0)
    # '''
    # 按tensor1和tensor2的降序去做
    # '''
    # diff2 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor2)))
    # diff1 = torch.cat((torch.ones(1, device=device1, dtype=torch.int32), torch.diff(tensor1)))
    # mask = (diff2 != 0) | (diff1 != 0)
    # return tensor1[mask], tensor2[mask]


def sp_edge_unique_mask(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    不适用off的segment_unique
    '''
    tensor2, ind1 = torch.sort(tensor2)
    tensor1 = tensor1[ind1]
    tensor1, ind2 = torch.sort(tensor1, stable=True)
    tensor2 = tensor2[ind2]
    diff2 = torch.cat((torch.ones(1, device=device, dtype=torch.int32), torch.diff(tensor2)))
    diff1 = torch.cat((torch.ones(1, device=device, dtype=torch.int32), torch.diff(tensor1)))
    mask = (diff2 != 0) | (diff1 != 0)
    return mask, ind1, ind2


# def sp_edge_unique_old(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     merge = tensor1 * off + tensor2
#     merge, inverse_indices = torch.unique(merge, return_inverse=True)

#     source_indices = torch.arange(0, tensor1.size(0), device=device, dtype=torch.long)
#     unique_indices = torch.zeros(merge.size(0), device=device, dtype=torch.long)

#     unique_indices.scatter_(0, inverse_indices, source_indices)

#     tensor1 = tensor1[unique_indices]
#     tensor2 = tensor2[unique_indices]

#     return tensor1, tensor2


# if __name__ == '__main__':
#     # tensor1 = torch.tensor([100, 199, 266, 100, 100, 299], device=device)
#     # tensor2 = torch.tensor([10, 155, 269, 100, 10, 37], device=device)
#     tensor1 = torch.randint(1, 3000, (800000,), device=device)
#     tensor2 = torch.randint(1, 3000, (800000,), device=device)

#     print(sp_edge_unique(tensor1, tensor2))

#     print(sp_edge_unique_old(tensor1, tensor2))

#     print(sp_edge_unique(tensor1, tensor2))

#     print(sp_edge_unique_old(tensor1, tensor2))
