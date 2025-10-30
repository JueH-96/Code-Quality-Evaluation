from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n - k + 1  # total number of subarrays of size k
        unique_nums = set(nums)  # set of unique values in nums
        max_almost_missing = -1  # initialize the result to -1
        
        for x in unique_nums:
            # find all positions where x appears
            pos = [i for i in range(n) if nums[i] == x]
            
            # compute the number of subarrays with no x
            num_no_x = 0
            
            # left gap length
            len_left = pos[0]
            num_no_x += max(0, len_left - k + 1)
            
            # between gaps
            for i in range(len(pos) - 1):
                len_between = pos[i + 1] - pos[i] - 1
                num_no_x += max(0, len_between - k + 1)
            
            # right gap length
            len_right = n - pos[-1] - 1
            num_no_x += max(0, len_right - k + 1)
            
            # number of subarrays that contain at least one x
            num_with_x = m - num_no_x
            
            # if x appears in exactly one subarray, update the max
            if num_with_x == 1:
                if x > max_almost_missing:
                    max_almost_missing = x
        
        return max_almost_missing