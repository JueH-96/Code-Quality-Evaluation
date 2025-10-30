from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Calculate prefix counts
        prefix_counts = [0] * n
        prefix_set = set()
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_counts[i] = len(prefix_set)
        
        # Calculate suffix counts
        suffix_counts = [0] * (n + 1)
        suffix_set = set()
        for i in range(n - 1, -1, -1):
            suffix_set.add(nums[i])
            suffix_counts[i] = len(suffix_set)
        
        # Calculate the difference array
        diff = [0] * n
        for i in range(n):
            diff[i] = prefix_counts[i] - suffix_counts[i + 1]
        
        return diff