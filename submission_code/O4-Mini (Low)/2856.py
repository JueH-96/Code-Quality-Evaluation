from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Total number of distinct elements in the entire array
        total_distinct = len(set(nums))
        
        result = 0
        # Try every starting index i
        for i in range(n):
            seen = set()
            # Expand the subarray from i to j
            for j in range(i, n):
                if nums[j] not in seen:
                    seen.add(nums[j])
                # Once we've seen all distinct elements, every extension
                # from j to end will also be complete
                if len(seen) == total_distinct:
                    result += (n - j)
                    break
        
        return result