class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute the cost for every possible substring to be a semi-palindrome
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
                        right = j - (length - 1 - mod) % d
                        while left < right:
                            if s[left] != s[right]:
                                current_cost += 1
                            left += d
                            right -= d
                    if current_cost < min_cost:
                        min_cost = current_cost
                cost[i][j] = min_cost
        
        # DP table to store the minimum cost for the first i characters to be divided into j parts
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for l in range(i):
                    if dp[l][j - 1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[l][j - 1] + cost[l][i - 1])
        
        return dp[n][k]