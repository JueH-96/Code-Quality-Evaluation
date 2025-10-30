class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        if not rewardValues:
            return 0
        
        unique_rewards = sorted(set(rewardValues))
        max_reward = unique_rewards[-1]
        n = 2 * max_reward + 1
        dp = [False] * n
        dp[0] = True
        
        for r in unique_rewards:
            for x in range(r-1, -1, -1):
                if dp[x]:
                    new_total = x + r
                    if new_total < n:
                        dp[new_total] = True
        
        for x in range(n-1, -1, -1):
            if dp[x]:
                return x
        
        return 0