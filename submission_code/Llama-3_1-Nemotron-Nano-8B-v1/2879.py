import math

class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        if n < 2 or k == 0:
            return 0
        
        # Precompute cost[i][j] for all substrings s[i..j] (i <= j)
        cost = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                m = j - i + 1
                divisors = set()
                # Find all divisors of m that are less than m
                for d in range(1, int(math.sqrt(m)) + 1):
                    if m % d == 0:
                        if d < m:
                            divisors.add(d)
                        other = m // d
                        if other < m and other != d:
                            divisors.add(other)
                min_cost = float('inf')
                for d in divisors:
                    total = 0
                    for g in range(d):
                        start = i + g
                        end = j
                        group = []
                        while start <= end:
                            group.append(s[start])
                            start += d
                        l = len(group)
                        cnt = 0
                        for p in range(l // 2):
                            if group[p] != group[l - 1 - p]:
                                cnt += 1
                        total += cnt
                    if total < min_cost:
                        min_cost = total
                cost[i][j] = min_cost
        
        # Dynamic Programming setup
        INF = float('inf')
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            for i in range(2 * j, n + 1):
                # Iterate over possible previous split points l
                for l in range(2 * (j - 1), i - 1):
                    if dp[l][j - 1] + cost[l][i - 1] < dp[i][j]:
                        dp[i][j] = dp[l][j - 1] + cost[l][i - 1]
        
        return dp[n][k] if dp[n][k] != INF else -1