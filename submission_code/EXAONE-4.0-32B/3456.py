class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [[0] * (k + 2) for _ in range(n)]
        
        for i in range(n):
            dp[i][1] = 1
            
        for i in range(n):
            for j in range(i):
                if nums[j] == nums[i]:
                    for g in range(1, k + 2):
                        if dp[j][g] != 0:
                            if dp[j][g] + 1 > dp[i][g]:
                                dp[i][g] = dp[j][g] + 1
                else:
                    for g in range(2, k + 2):
                        if dp[j][g - 1] != 0:
                            if dp[j][g - 1] + 1 > dp[i][g]:
                                dp[i][g] = dp[j][g - 1] + 1
        
        ans = 0
        for i in range(n):
            for g in range(1, k + 2):
                if dp[i][g] > ans:
                    ans = dp[i][g]
        return ans