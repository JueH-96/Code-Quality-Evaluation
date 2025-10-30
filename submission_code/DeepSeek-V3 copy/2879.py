class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute the cost for every possible substring
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length == 1:
                    cost[i][j] = float('inf')
                    continue
                min_cost = float('inf')
                for d in range(1, length):
                    if length % d != 0:
                        continue
                    current_cost = 0
                    for mod in range(d):
                        left = i + mod
                        right = j - (d - 1 - mod)
                        while left < right:
                            if s[left] != s[right]:
                                current_cost += 1
                            left += d
                            right -= d
                    if current_cost < min_cost:
                        min_cost = current_cost
                cost[i][j] = min_cost
        
        # DP to find the minimum total cost for k partitions
        dp = [[float('inf')] * (k + 1) for _ in range(n)]
        for i in range(n):
            dp[i][1] = cost[0][i]
        for j in range(2, k + 1):
            for i in range(n):
                for m in range(j - 2, i):
                    if dp[m][j - 1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[m][j - 1] + cost[m + 1][i])
        
        return dp[n - 1][k]