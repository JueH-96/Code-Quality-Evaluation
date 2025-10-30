class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        unique_rewards = sorted(set(rewardValues))
        sum_max = sum(unique_rewards)
        achievable = [False] * (sum_max + 1)
        achievable[0] = True
        
        for r in unique_rewards:
            for x in range(r - 1, -1, -1):
                if achievable[x]:
                    achievable[x + r] = True
        
        # Find the maximum achievable x
        for x in range(sum_max, -1, -1):
            if achievable[x]:
                return x
        return 0