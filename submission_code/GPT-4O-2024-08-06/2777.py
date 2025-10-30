from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_distinct = [0] * n
        suffix_distinct = [0] * n
        
        # Calculate distinct elements in prefix for each index
        seen = set()
        for i in range(n):
            seen.add(nums[i])
            prefix_distinct[i] = len(seen)
        
        # Calculate distinct elements in suffix for each index
        seen.clear()
        for i in range(n-1, -1, -1):
            seen.add(nums[i])
            suffix_distinct[i] = len(seen)
        
        # Calculate the distinct difference array
        diff = [0] * n
        for i in range(n):
            suffix_count = suffix_distinct[i+1] if i+1 < n else 0
            diff[i] = prefix_distinct[i] - suffix_count
        
        return diff