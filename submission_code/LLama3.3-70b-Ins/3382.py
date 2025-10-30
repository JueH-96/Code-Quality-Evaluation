from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                max_val = max(subarray)
                if subarray[0] == max_val and subarray[-1] == max_val:
                    count += 1
        return count