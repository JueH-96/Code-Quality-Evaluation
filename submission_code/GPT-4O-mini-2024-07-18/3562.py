from typing import List
from itertools import combinations

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        intervals = sorted((l, r, w, i) for i, (l, r, w) in enumerate(intervals))
        
        # To store the best score and the corresponding indices
        best_score = 0
        best_indices = []

        # Check combinations of up to 4 intervals
        for k in range(1, 5):
            for comb in combinations(range(n), k):
                total_weight = sum(intervals[i][2] for i in comb)
                if self.is_non_overlapping(comb, intervals):
                    if (total_weight > best_score) or (total_weight == best_score and list(comb) < best_indices):
                        best_score = total_weight
                        best_indices = list(comb)

        # Return the indices sorted in ascending order
        return sorted(best_indices)

    def is_non_overlapping(self, comb: tuple, intervals: List[List[int]]) -> bool:
        last_end = -1
        for i in comb:
            l, r, _, _ = intervals[i]
            if l <= last_end:  # Overlapping condition
                return False
            last_end = r
        return True