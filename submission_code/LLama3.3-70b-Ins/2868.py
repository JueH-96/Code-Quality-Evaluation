from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if all(abs(subarray[k] - subarray[k+1]) <= 2 for k in range(len(subarray)-1)):
                    res += 1
                    
        return res