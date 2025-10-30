from typing import List

class FenwickTree:
    def __init__(self, size):
        self.N = size + 2  # 1-indexed
        self.tree = [0] * (self.N)
    
    def update(self, index, value):
        while index < self.N:
            if self.tree[index] < value:
                self.tree[index] = value
            else:
                break  # No need to update further if value is not greater
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        points = [ (x, y, idx) for idx, (x, y) in enumerate(coordinates) ]
        
        # For dp_end
        sorted_points_end = sorted(points, key=lambda p: (p[0], p[1]))
        unique_y_end = sorted(set(p[1] for p in points))
        y_rank_end = {y: i+1 for i, y in enumerate(unique_y_end)}
        
        m_end = len(unique_y_end)
        ft_end = FenwickTree(m_end)
        dp_end = [0] * len(points)
        
        for p in sorted_points_end:
            y = p[1]
            idx = p[2]
            rank = y_rank_end[y]
            max_before = ft_end.query(rank - 1)
            dp_end[idx] = max_before + 1
            ft_end.update(rank, dp_end[idx])
        
        # For dp_start
        sorted_points_start = sorted(points, key=lambda p: (-p[0], -p[1]))
        unique_y_start = sorted(set(p[1] for p in points), reverse=True)
        y_rank_start = {y: i+1 for i, y in enumerate(unique_y_start)}
        
        m_start = len(unique_y_start)
        ft_start = FenwickTree(m_start)
        dp_start = [0] * len(points)
        
        for p in sorted_points_start:
            y = p[1]
            idx = p[2]
            rank = y_rank_start[y]
            max_before = ft_start.query(rank - 1)
            dp_start[idx] = max_before + 1
            ft_start.update(rank, dp_start[idx])
        
        # Combine for point k
        return dp_end[k] + dp_start[k] - 1