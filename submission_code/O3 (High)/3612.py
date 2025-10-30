from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # it is impossible to have two adjacent sub-arrays of length k
        # if the total length is smaller than 2 * k
        if 2 * k > n:
            return False
        
        # helper: check whether nums[start .. start+k-1] is strictly increasing
        def strictly_increasing(start: int) -> bool:
            for i in range(start, start + k - 1):
                if nums[i] >= nums[i + 1]:
                    return False
            return True
        
        # slide a window of size 2 * k across the array
        for start in range(0, n - 2 * k + 1):
            if strictly_increasing(start) and strictly_increasing(start + k):
                return True
        
        return False