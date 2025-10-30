class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        ans = 1
        last = nums[-1]
        
        for i in range(n - 2, -1, -1):
            if nums[i] <= last:
                ans += 1
                last = nums[i]
            else:
                last = nums[i]
        
        return ans