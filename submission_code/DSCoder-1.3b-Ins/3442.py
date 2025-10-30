class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        dp = [0] * (1 << n)
        sums = [0] * (1 << n)
        for mask in range(1, 1 << n):
            for i in range(n):
                if ((mask >> i) & 1):
                    sums[mask] += rewardValues[i]
                    next_mask = mask ^ (1 << i)
                    if next_mask in sums:
                        dp[mask] = max(dp[mask], sums[mask] + dp[next_mask])
        return max(dp)