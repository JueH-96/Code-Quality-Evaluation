class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        dp = {0: 0}
        for r in rewardValues:
            tmp = {}
            for s in list(dp.keys()):
                if r > s:
                    new_sum = s + r
                    new_reward = dp[s] + r
                    if new_sum in tmp:
                        if new_reward > tmp[new_sum]:
                            tmp[new_sum] = new_reward
                    else:
                        tmp[new_sum] = new_reward
            for s_new in tmp:
                if s_new in dp:
                    if tmp[s_new] > dp[s_new]:
                        dp[s_new] = tmp[s_new]
                else:
                    dp[s_new] = tmp[s_new]
        return max(dp.values()) if dp else 0