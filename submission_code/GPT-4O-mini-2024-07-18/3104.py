class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        
        count = 0
        
        for i in range(n + 1):
            if i == 0 or i > nums[i - 1]:
                count += 1
        
        return count