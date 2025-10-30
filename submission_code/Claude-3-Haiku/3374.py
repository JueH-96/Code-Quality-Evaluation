class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                if self.isAlternating(nums, i, j):
                    count += 1
        
        return count
    
    def isAlternating(self, nums: List[int], start: int, end: int) -> bool:
        prev = -1
        for i in range(start, end+1):
            if nums[i] == prev:
                return False
            prev = nums[i]
        return True