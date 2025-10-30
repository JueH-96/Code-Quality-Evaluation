from typing import List
from itertools import combinations

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by their end time to facilitate finding non-overlapping intervals
        intervals = sorted(enumerate(intervals), key=lambda x: x[1][1])
        
        def is_non_overlapping(selected_intervals):
            # Check if the selected intervals are non-overlapping
            for i in range(len(selected_intervals) - 1):
                if intervals[selected_intervals[i]][1][1] >= intervals[selected_intervals[i + 1]][1][0]:
                    return False
            return True
        
        max_weight = 0
        best_combination = []
        
        # Try all combinations of up to 4 intervals
        for k in range(1, 5):
            for comb in combinations(range(len(intervals)), k):
                if is_non_overlapping(comb):
                    current_weight = sum(intervals[i][1][2] for i in comb)
                    if current_weight > max_weight:
                        max_weight = current_weight
                        best_combination = comb
                    elif current_weight == max_weight:
                        # Check lexicographical order
                        if list(comb) < best_combination:
                            best_combination = comb
        
        # Return the indices of the best combination
        return sorted([intervals[i][0] for i in best_combination])