class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n)]
        ans = 1
        for j in range(n):
            dp[j][0] = 1
            for i in range(j):
                if nums[i] == nums[j]:
                    for t in range(k + 1):
                        if dp[i][t] == 0:
                            continue
                        new_val = dp[i][t] + 1
                        if new_val > dp[j][t]:
                            dp[j][t] = new_val
                            if new_val > ans:
                                ans = new_val
                else:
                    for t in range(k):
                        if dp[i][t] == 0:
                            continue
                        new_val = dp[i][t] + 1
                        if new_val > dp[j][t + 1]:
                            dp[j][t + 1] = new_val
                            if new_val > ans:
                                ans = new_val
        return ans