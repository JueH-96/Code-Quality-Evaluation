from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [0] * n
        seen_prefix = set()
        
        # Compute prefix distinct counts
        for i in range(n):
            seen_prefix.add(nums[i])
            prefix[i] = len(seen_prefix)
        
        suffix = [0] * n
        seen_suffix = set()
        
        # Compute suffix distinct counts from right to left
        for i in range(n - 1, -1, -1):
            seen_suffix.add(nums[i])
            suffix[i] = len(seen_suffix)
        
        result = [0] * n
        
        # Calculate the distinct difference array: prefix distinct - suffix distinct (for the suffix starting at i+1)
        for i in range(n):
            if i == n - 1:
                # Suffix is empty for the last element, so subtract 0
                result[i] = prefix[i]
            else:
                result[i] = prefix[i] - suffix[i + 1]
        
        return result