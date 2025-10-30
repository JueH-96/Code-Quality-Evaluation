from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def matches_pattern(subarray, pattern):
            for k in range(len(pattern)):
                if pattern[k] == 1 and not (subarray[k + 1] > subarray[k]):
                    return False
                elif pattern[k] == 0 and not (subarray[k + 1] == subarray[k]):
                    return False
                elif pattern[k] == -1 and not (subarray[k + 1] < subarray[k]):
                    return False
            return True

        count = 0
        m = len(pattern)
        for i in range(len(nums) - m):
            subarray = nums[i:i + m + 1]
            if matches_pattern(subarray, pattern):
                count += 1

        return count