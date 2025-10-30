from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_elements = len(set(nums))
        count = 0
        for i in range(len(nums)):
            seen = set()
            for j in range(i, len(nums)):
                seen.add(nums[j])
                if len(seen) == distinct_elements:
                    count += 1
        return count