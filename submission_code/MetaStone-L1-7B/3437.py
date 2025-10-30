class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from bisect import bisect_right
        
        count = Counter(power)
        sorted_x = sorted(count.keys())
        dp = []
        for i in range(len(sorted_x)):
            x = sorted_x[i]
            target = x - 3
            j = bisect_right(sorted_x, target) - 1
            if j >= 0:
                take = x * count[x] + dp[j]
            else:
                take = x * count[x]
            not_take = dp[i-1] if i > 0 else 0
            dp.append(max(take, not_take))
        return dp[-1]