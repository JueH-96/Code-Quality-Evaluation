from typing import List
from itertools import combinations

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        def is_non_overlapping(intervals):
            intervals.sort(key=lambda x: x[0])
            for i in range(1, len(intervals)):
                if intervals[i][0] <= intervals[i-1][1]:
                    return False
            return True
        
        n = len(intervals)
        max_weight = 0
        best_combination = []
        
        for k in range(1, 5):
            for comb in combinations(range(n), k):
                selected_intervals = [intervals[i] for i in comb]
                if is_non_overlapping(selected_intervals):
                    current_weight = sum(intervals[i][2] for i in comb)
                    if current_weight > max_weight or (current_weight == max_weight and list(comb) < best_combination):
                        max_weight = current_weight
                        best_combination = list(comb)
        
        return best_combination

# Example usage:
# sol = Solution()
# print(sol.maximumWeight([[1,3,2],[4,5,2],[1,5,5],[6,9,3],[6,7,1],[8,9,1]]))  # Output: [2, 3]
# print(sol.maximumWeight([[5,8,1],[6,7,7],[4,7,3],[9,10,6],[7,8,2],[11,14,3],[3,5,5]]))  # Output: [1, 3, 5, 6]