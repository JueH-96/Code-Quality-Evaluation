from typing import List
import bisect

class FenwickTree:
    def __init__(self, size):
        self.N = size + 2
        self.tree = [0] * (self.N + 1)
    
    def update(self, index, value):
        while index < self.N:
            if value > self.tree[index]:
                self.tree[index] = value
            index += index & -index
    
    def query(self, index):
        res = float('-inf')
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        if not intervals:
            return []
        
        n = len(intervals)
        # Sort intervals by end time
        intervals.sort(key=lambda x: x[1])
        sorted_ends = [interval[1] for interval in intervals]
        
        # Compress coordinates
        unique_ends = sorted(set(sorted_ends))
        end_to_index = {end: i + 1 for i, end in enumerate(unique_ends)}
        
        # Initialize Fenwick trees for up to 4 selections
        fenwick_trees = [FenwickTree(len(unique_ends)) for _ in range(4)]
        
        # Initialize DP and prev arrays
        DP = [[0] * 5 for _ in range(n)]
        prev = [[-1] * 5 for _ in range(n)]
        
        max_weight = 0
        best_k = 0
        best_i = -1
        
        for j in range(n):
            start_j = intervals[j][0]
            weight_j = intervals[j][2]
            end_j_index = end_to_index[intervals[j][1]]
            
            for k in range(1, 5):
                if k == 1:
                    DP[j][k] = weight_j
                    prev[j][k] = -1
                else:
                    # Find the maximum DP[k-1][m] where end[m] < start_j
                    end_m_index = bisect.bisect_left(unique_ends, start_j) - 1
                    if end_m_index >= 0:
                        query_value = fenwick_trees[k-2].query(end_to_index[unique_ends[end_m_index]])
                        if query_value != float('-inf'):
                            DP[j][k] = query_value + weight_j
                            m = bisect.bisect_left(sorted_ends, unique_ends[end_m_index])
                            prev[j][k] = m
                        else:
                            DP[j][k] = weight_j
                            prev[j][k] = -1
                    else:
                        DP[j][k] = weight_j
                        prev[j][k] = -1
                # Update Fenwick tree for k selections
                fenwick_trees[k-1].update(end_j_index, DP[j][k])
                # Track the best DP value
                if DP[j][k] > max_weight:
                    max_weight = DP[j][k]
                    best_k = k
                    best_i = j
        
        # Backtrack to find the selected intervals
        selected = []
        k = best_k
        i = best_i
        while k > 0:
            if prev[i][k] != -1:
                m = prev[i][k]
                selected.append(i)
                i = m
                k -= 1
            else:
                selected.append(i)
                break
        
        # Map to original indices and sort
        original_indices = [intervals[idx][3] for idx in selected]
        original_indices.sort()
        
        return original_indices