from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        """
        For every index i compute:
            diff[i] = (# distinct elements in nums[0..i]) -
                      (# distinct elements in nums[i+1..n-1])
        """
        n = len(nums)
        
        # prefix_counts[i] -> distinct count in nums[0..i]
        prefix_counts = [0] * n
        seen = set()
        for i, val in enumerate(nums):
            seen.add(val)
            prefix_counts[i] = len(seen)
        
        # suffix_counts[i] -> distinct count in nums[i..n-1]
        # we actually need suffix starting from i+1, so compute counts for each
        suffix_counts = [0] * n
        seen.clear()
        for i in range(n - 1, -1, -1):
            suffix_counts[i] = len(seen)          # suffix after index i
            seen.add(nums[i])
        
        # build result
        return [prefix_counts[i] - suffix_counts[i] for i in range(n)]