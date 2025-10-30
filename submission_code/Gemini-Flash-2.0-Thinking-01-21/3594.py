import itertools
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        max_outlier = -float('inf')
        indices = list(range(n))
        for special_indices_tuple in itertools.combinations(indices, n - 2):
            special_indices = list(special_indices_tuple)
            special_numbers = [nums[i] for i in special_indices]
            special_sum = sum(special_numbers)
            remaining_indices = []
            for i in indices:
                if i not in special_indices:
                    remaining_indices.append(i)
            remaining_numbers = [nums[i] for i in remaining_indices]
            if special_sum in remaining_numbers:
                for num in remaining_numbers:
                    if num != special_sum:
                        max_outlier = max(max_outlier, num)
        return max_outlier