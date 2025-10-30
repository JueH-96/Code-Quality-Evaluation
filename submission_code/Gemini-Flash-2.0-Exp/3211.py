class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        last = float('-inf')
        curr = 0
        for i in range(n):
            curr += nums[i]
            if curr >= last:
                ans += 1
                last = curr
                curr = 0
            
        if curr > 0:
            ans += 1
        return ans