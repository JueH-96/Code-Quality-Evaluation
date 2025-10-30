import math
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx, val):
        while idx <= self.size:
            self.tree[idx] = max(self.tree[idx], val)
            idx += idx & -idx
    
    def query(self, idx):
        if idx < 1:
            return 0
        res = 0
        i = idx
        while i > 0:
            res = max(res, self.tree[i])
            i -= i & -i
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # Get unique y values and assign ranks
        y_vals = sorted(set(coord[1] for coord in coordinates))
        m = len(y_vals)
        y_to_rank = {y: idx + 1 for idx, y in enumerate(y_vals)}  # rank from 1 to m
        
        # Create points list with x, y, orig_idx
        points = [(coord[0], coord[1], i) for i, coord in enumerate(coordinates)]
        
        # Compute end_len
        ft_end = FenwickTree(m)
        end_len_list = [0] * n
        sorted_points_end = sorted(points, key=lambda p: p[0])  # sort by x asc
        for x, y, idx in sorted_points_end:
            r = y_to_rank[y]
            max_prev = ft_end.query(r - 1)  # prefix max up to r-1
            end_len_val = max_prev + 1
            end_len_list[idx] = end_len_val
            ft_end.update(r, end_len_val)
        
        # Compute start_len
        ft_start = FenwickTree(m)
        start_len_list = [0] * n
        sorted_points_start = sorted(points, key=lambda p: p[0], reverse=True)  # sort by x desc
        for x, y, idx in sorted_points_start:
            r = y_to_rank[y]  # y rank
            val = m + 1 - r  # y_rank_rev index
            query_idx = val - 1
            max_prev = ft_start.query(query_idx)  # prefix max up to query_idx in y_rank_rev space
            start_len_val = max_prev + 1  # max_prev is start_len of q or 0
            start_len_list[idx] = start_len_val
            ft_start.update(val, start_len_val)  # update at y_rank_rev index
        
        # The longest path through k is end_len + start_len - 1
        ans = end_len_list[k] + start_len_list[k] - 1
        return ans