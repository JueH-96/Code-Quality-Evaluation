class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        ans = 0
        if nums[0] > 0:
            ans += 1
        
        for i in range(n):
            if i < n - 1 and nums[i] < i + 1 and nums[i+1] > i + 1:
                ans += 1
            elif i == n - 1 and nums[i] < i + 1:
                ans += 1
        return ans