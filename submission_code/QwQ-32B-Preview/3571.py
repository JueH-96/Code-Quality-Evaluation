from typing import List
from sortedcontainers import SortedList

class FenwickTree:
    def __init__(self, size):
        self.N = size + 5
        self.tree = [0] * self.N
    
    def update(self, index, value):
        while index < self.N:
            self.tree[index] = max(self.tree[index], value)
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res = max(res, self.tree[index])
            index -= index & -index
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # Sort coordinates by x ascending
        coordinates_sorted = sorted enumerate(coordinates, key=lambda x: x[1][0])
        
        # Compress y coordinates
        y_values = SortedList([coord[1] for coord in coordinates_sorted])
        y_index = {y: idx + 1 for idx, y in enumerate(y_values)}
        
        # Forward pass: longest increasing path ending at each point
        forward = [0] * n
        ft_forward = FenwickTree(n)
        for idx, coord in coordinates_sorted:
            y = y_index[coord[1]]
            max_len = ft_forward.query(y - 1) + 1
            forward[idx] = max_len
            ft_forward.update(y, max_len)
        
        # Sort coordinates by x descending
        coordinates_sorted_desc = sorted enumerate(coordinates, key=lambda x: -x[1][0])
        
        # Compress y coordinates for backward pass
        y_values_desc = SortedList([coord[1] for idx, coord in coordinates_sorted_desc])
        y_index_desc = {y: idx + 1 for idx, y in enumerate(y_values_desc)}
        
        # Backward pass: longest increasing path starting from each point
        backward = [0] * n
        ft_backward = FenwickTree(n)
        for idx, coord in coordinates_sorted_desc:
            y = y_index_desc[coord[1]]
            max_len = ft_backward.query(y - 1) + 1
            backward[idx] = max_len
            ft_backward.update(y, max_len)
        
        # Combine forward and backward for point k
        total = forward[k] + backward[k] - 1
        return total