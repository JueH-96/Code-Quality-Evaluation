from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # total number of distinct elements in the entire array
        k = len(set(nums))
        n = len(nums)
        count = 0
        
        # enumerate starting index of the subarray
        for i in range(n):
            seen = set()
            # extend the subarray from i to j
            for j in range(i, n):
                seen.add(nums[j])
                # if the distinct elements collected so far equals the overall distinct count, it's complete
                if len(seen) == k:
                    count += 1
        return count