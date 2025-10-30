class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute the cost to make substrings semi-palindromes
        cost = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                min_cost = float('inf')
                for d in range(1, len(sub)):
                    if len(sub) % d == 0:
                        temp_cost = 0
                        for m in range(d):
                            temp = sub[m::d]
                            temp_cost += sum(temp[i] != temp[-i-1] for i in range(len(temp) // 2))
                        min_cost = min(min_cost, temp_cost)
                cost[i][j] = min_cost
        
        # Dynamic programming to find the minimum cost partition
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for x in range(j - 1, i):
                    dp[i][j] = min(dp[i][j], dp[x][j - 1] + cost[x][i - 1])
        
        return dp[n][k]