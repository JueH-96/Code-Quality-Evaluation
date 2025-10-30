class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        target = n
        dp = [float('inf')] * (target + 1)
        dp[0] = 0

        for i in range(n):
            t = time[i] + 1
            c = cost[i]
            for v in range(target, -1, -1):
                new_v = min(v + t, target)
                if dp[v] + c < dp[new_v]:
                    dp[new_v] = dp[v] + c

        return dp[target]