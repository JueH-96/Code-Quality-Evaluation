from typing import List
import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [-float('inf')] * (self.n + 1)  # 1-based indexing
    
    def update(self, idx, value):
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
                idx += idx & -idx
            else:
                break
    
    def query(self, idx):
        res = -float('inf')
        while idx > 0:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from itertools import groupby
        
        n = len(coordinates)
        if n == 0:
            return 0
        
        # Preprocess unique_ys and y_to_original_rank
        ys = [y for x, y in coordinates]
        unique_ys = sorted(list(set(ys)))
        y_to_original_rank = {y: i+1 for i, y in enumerate(unique_ys)}  # 1-based
        m = len(unique_ys)
        
        # Compute dp_end
        sorted_dp_end = sorted([(coordinates[i][0], coordinates[i][1], i) for i in range(n)], key=lambda p: (p[0], p[1]))
        dp_end = [0] * n
        ft_end = FenwickTree(m)
        i = 0
        while i < n:
            current_x = sorted_dp_end[i][0]
            j = i
            while j < n and sorted_dp_end[j][0] == current_x:
                j += 1
            # Process group i..j-1
            temp = []
            for idx in range(i, j):
                x, y, original_idx = sorted_dp_end[idx]
                original_rank = y_to_original_rank[y]
                max_prev = ft_end.query(original_rank - 1)
                if max_prev == -float('inf'):
                    max_prev = 0
                current_dp = max_prev + 1
                dp_end[original_idx] = current_dp
                temp.append((original_rank, current_dp))
            # Update Fenwick Tree
            for original_rank, current_dp in temp:
                ft_end.update(original_rank, current_dp)
            i = j
        
        # Compute dp_start
        sorted_dp_start = sorted([(coordinates[i][0], coordinates[i][1], i) for i in range(n)], key=lambda p: (-p[0], -p[1]))
        dp_start = [0] * n
        ft_start = FenwickTree(m)
        i = 0
        while i < n:
            current_x = sorted_dp_start[i][0]
            j = i
            while j < n and sorted_dp_start[j][0] == current_x:
                j += 1
            # Process group i..j-1
            temp = []
            for idx in range(i, j):
                x, y, original_idx = sorted_dp_start[idx]
                original_rank = y_to_original_rank[y]
                reversed_rank = m - original_rank + 1
                max_next = ft_start.query(reversed_rank - 1)
                if max_next == -float('inf'):
                    max_next = 0
                current_dp = max_next + 1
                dp_start[original_idx] = current_dp
                temp.append((reversed_rank, current_dp))
            # Update Fenwick Tree
            for reversed_rank, current_dp in temp:
                ft_start.update(reversed_rank, current_dp)
            i = j
        
        # Return result for k
        return dp_end[k] + dp_start[k] - 1