import torch
from torch_scatter import segment_csr

class TrussFile(object):
    def __init__(self, row_ptr: torch.Tensor = None, columns: torch.Tensor = None,
                 truss_result: torch.Tensor = None, old_vertices_hash: torch.Tensor = None,
                 source_edge: torch.Tensor = None, left_edge: torch.Tensor = None, right_edge: torch.Tensor = None,):
        self.row_ptr = row_ptr
        self.columns = columns
        self.truss_result = truss_result
        self.old_vertices_hash = old_vertices_hash
        self.source_edge = source_edge
        self.left_edge = left_edge
        self.right_edge = right_edge

    def save(self, filename):
        # 保存对象的状态字典（包含所有GPU上的Tensor）
        torch.save(self.__dict__, filename)

    def load(self, filename: str):
        # 加载状态字典并恢复对象状态
        state_dict = torch.load(filename)
        self.__dict__.update(state_dict)

    def get_all_data(self):
        return self.row_ptr, self.columns, self.truss_result, self.old_vertices_hash, self.source_edge, self.left_edge, self.right_edge

    def get_com_data(self):
        return self.row_ptr, self.columns, self.truss_result, self.old_vertices_hash


if __name__ == '__main__':
    truss_file = TrussFile()
    truss_file.load("dblp_no.pt")

    print(truss_file.row_ptr)