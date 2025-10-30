class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        dp = {}
        dp[0] = 0
        for num in rewardValues:
            current_sums = list(dp.keys())
            for x in current_sums:
                if num > x:
                    new_sum = x + num
                    if new_sum in dp:
                        if dp[x] + num > dp[new_sum]:
                            dp[new_sum] = dp[x] + num
                    else:
                        dp[new_sum] = dp[x] + num
        return max(dp.values()) if dp else 0