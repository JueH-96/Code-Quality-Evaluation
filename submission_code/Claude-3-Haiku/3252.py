class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                if self.isIncremovable(nums, i, j):
                    count += 1
        
        return count
    
    def isIncremovable(self, nums: List[int], start: int, end: int) -> bool:
        for i in range(start, end):
            if nums[i] >= nums[i+1]:
                return False
        return True