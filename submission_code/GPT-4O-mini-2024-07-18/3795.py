from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        
        # Function to apply queries up to the k-th index
        def apply_queries(k: int) -> List[int]:
            temp_nums = nums[:]
            for i in range(k):
                l, r, val = queries[i]
                for j in range(l, r + 1):
                    temp_nums[j] -= val
            return temp_nums
        
        # Check for the minimum k that makes nums a zero array
        for k in range(1, len(queries) + 1):
            modified_nums = apply_queries(k)
            if all(x == 0 for x in modified_nums):
                return k
        
        return -1