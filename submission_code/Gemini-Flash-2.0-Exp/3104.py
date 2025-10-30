class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        
        # Case 1: Select no student
        if nums[0] > 0:
            count += 1
        
        # Case 2: Select some students
        for i in range(n - 1):
            if nums[i] <= i and nums[i+1] > i + 1:
                count += 1
        
        # Case 3: Select all students
        if nums[n-1] < n:
            count += 1
            
        return count