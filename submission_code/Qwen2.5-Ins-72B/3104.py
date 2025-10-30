from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Check if selecting 0 students is a valid way
        if nums[0] > 0:
            count += 1
        
        # Check for valid ways by selecting k students
        for i in range(n - 1):
            if nums[i] < i + 1 and nums[i + 1] > i + 1:
                count += 1
        
        # Check if selecting all students is a valid way
        if nums[-1] < n:
            count += 1
        
        return count