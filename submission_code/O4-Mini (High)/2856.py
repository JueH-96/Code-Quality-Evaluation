from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Total distinct elements in the whole array
        distinct_total = len(set(nums))
        n = len(nums)
        result = 0
        
        # For each start index i, expand until we've seen all distinct elements
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                if len(seen) == distinct_total:
                    # Once we've hit all distinct elements, every further extension
                    # from j to end of array is also complete
                    result += (n - j)
                    break
        
        return result