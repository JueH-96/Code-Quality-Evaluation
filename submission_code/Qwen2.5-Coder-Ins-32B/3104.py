from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ways = 0
        
        # Check if selecting no students is a valid way
        if nums[0] > 0:
            ways += 1
        
        # Check for each possible number of selected students
        for i in range(n):
            if (i + 1 > nums[i]) and (i == n - 1 or i + 1 < nums[i + 1]):
                ways += 1
        
        # Check if selecting all students is a valid way
        if nums[-1] < n:
            ways += 1
        
        return ways