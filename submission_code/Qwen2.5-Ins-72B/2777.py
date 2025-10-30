from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_count = [0] * n
        suffix_count = [0] * n
        
        # Calculate prefix distinct counts
        prefix_set = set()
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_count[i] = len(prefix_set)
        
        # Calculate suffix distinct counts
        suffix_set = set()
        for i in range(n-1, -1, -1):
            suffix_count[i] = len(suffix_set)
            suffix_set.add(nums[i])
        
        # Calculate the distinct difference array
        diff = [prefix_count[i] - suffix_count[i] for i in range(n)]
        
        return diff