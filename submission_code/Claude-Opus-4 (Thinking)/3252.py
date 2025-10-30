class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                # Check if removing nums[i:j+1] results in strictly increasing array
                if self.isIncremovable(nums, i, j):
                    count += 1
        
        return count
    
    def isIncremovable(self, nums, start, end):
        n = len(nums)
        
        # Check left part (before start)
        for k in range(1, start):
            if nums[k] <= nums[k-1]:
                return False
        
        # Check right part (after end)
        for k in range(end + 2, n):
            if nums[k] <= nums[k-1]:
                return False
        
        # Check connection between left and right parts
        if start > 0 and end < n - 1:
            if nums[start-1] >= nums[end+1]:
                return False
        
        return True