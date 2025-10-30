from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = []
        seen_prefix = set()
        # Compute the number of distinct elements from the prefix (0..i)
        for num in nums:
            seen_prefix.add(num)
            prefix.append(len(seen_prefix))
            
        suffix_count = [0] * n
        seen_suffix = set()
        # Compute the number of distinct elements in the suffix (i+1..n-1)
        for i in range(n - 1, -1, -1):
            suffix_count[i] = len(seen_suffix)
            seen_suffix.add(nums[i])
            
        # Build the distinct difference array: prefix distinct count - suffix distinct count
        diff = []
        for i in range(n):
            diff.append(prefix[i] - suffix_count[i])
            
        return diff