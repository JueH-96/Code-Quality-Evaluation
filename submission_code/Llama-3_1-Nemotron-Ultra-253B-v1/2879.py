import math

class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = float('inf')
        
        # Precompute the cost for each substring [i..j]
        cost = [[INF] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                L = j - i + 1
                if L < 2:
                    continue
                min_changes = INF
                divisors = self.get_divisors(L)
                for d in divisors:
                    current = 0
                    for r in range(d):
                        group = list(range(r, L, d))
                        m = len(group)
                        for k_group in range(m // 2):
                            x = group[k_group]
                            y = group[m - 1 - k_group]
                            if s[i + x] != s[i + y]:
                                current += 1
                    if current < min_changes:
                        min_changes = current
                cost[i][j] = min_changes
        
        # Initialize DP table
        dp = [[INF] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        
        for m in range(1, k + 1):
            for i in range(2 * m, n + 1):
                for j in range(2 * (m - 1), i - 1):
                    if dp[m-1][j] + cost[j][i-1] < dp[m][i]:
                        dp[m][i] = dp[m-1][j] + cost[j][i-1]
        
        return dp[k][n] if dp[k][n] != INF else -1
    
    def get_divisors(self, L):
        divisors = set()
        for d in range(1, int(math.isqrt(L)) + 1):
            if L % d == 0:
                if d < L:
                    divisors.add(d)
                other = L // d
                if other != d and other < L:
                    divisors.add(other)
        return sorted(divisors)