class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        distinct = sorted(set(rewardValues))
        if not distinct:
            return 0
        max_val = max(distinct)
        up_bound = 2 * max_val
        dp = [False] * up_bound
        dp[0] = True
        
        for r in distinct:
            for x in range(r):
                if dp[x] and x + r < up_bound:
                    dp[x + r] = True
        
        for x in range(up_bound - 1, -1, -1):
            if dp[x]:
                return x
        return 0