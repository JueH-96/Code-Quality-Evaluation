class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        # Precompute the cost for each substring s[i..j]
        cost = [[0] * n for _ in range(n)]
        
        def get_divisors(m):
            divs = set()
            for d in range(1, int(m**0.5) + 1):
                if m % d == 0:
                    if d < m:
                        divs.add(d)
                    counterpart = m // d
                    if counterpart < m and counterpart != d:
                        divs.add(counterpart)
            return list(divs)
        
        for i in range(n):
            for j in range(i + 1, n):
                m = j - i + 1
                divisors = get_divisors(m)
                min_changes = float('inf')
                for d in divisors:
                    total = 0
                    for g in range(d):
                        group = []
                        pos = i + g
                        while pos <= j:
                            group.append(s[pos])
                            pos += d
                        l = len(group)
                        changes = 0
                        for start in range(l // 2):
                            end = l - 1 - start
                            if group[start] != group[end]:
                                changes += 1
                        total += changes
                    if total < min_changes:
                        min_changes = total
                cost[i][j] = min_changes
        
        # Dynamic programming part
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for p in range(1, k + 1):
                if p == 1:
                    if i >= 2:
                        dp[i][1] = cost[0][i-1]
                    else:
                        continue
                else:
                    start_m = 2 * (p - 1)
                    end_m = i - 2
                    if start_m > end_m:
                        continue
                    for m in range(start_m, end_m + 1):
                        if dp[m][p-1] + cost[m][i-1] < dp[i][p]:
                            dp[i][p] = dp[m][p-1] + cost[m][i-1]
        
        return dp[n][k]