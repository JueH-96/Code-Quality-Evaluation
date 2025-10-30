class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-1] * n
        dp[0] = 0
        
        for i in range(1, n):
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                if diff <= target:
                    if dp[j] != -1:
                        if dp[i] < dp[j] + 1:
                            dp[i] = dp[j] + 1
        return dp[-1]