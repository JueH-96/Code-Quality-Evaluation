class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        ans = 0
        current_and = nums[0]
        
        for i in range(n):
            current_and &= nums[i]
            if current_and == 0:
                ans += 1
                if i + 1 < n:
                    current_and = nums[i+1]
        
        if ans == 0:
            return 1
        else:
            return ans