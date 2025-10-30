from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def matches_pattern(subarray, pattern):
            for i, p in enumerate(pattern):
                if p == 1 and subarray[i + 1] <= subarray[i]:
                    return False
                elif p == 0 and subarray[i + 1] != subarray[i]:
                    return False
                elif p == -1 and subarray[i + 1] >= subarray[i]:
                    return False
            return True
        
        count = 0
        m = len(pattern)
        for i in range(len(nums) - m):
            if matches_pattern(nums[i:i + m + 1], pattern):
                count += 1
        
        return count