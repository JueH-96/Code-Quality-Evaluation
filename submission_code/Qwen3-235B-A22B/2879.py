class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        if n < 2 or k < 1:
            return 0
        
        INF = float('inf')
        # Precompute the cost matrix
        cost = [[INF for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                m = j - i + 1
                if m < 2:
                    continue
                min_cost = INF
                for d in range(1, m):
                    if m % d != 0:
                        continue
                    total = 0
                    for r in range(d):
                        first_pos = i + r
                        if first_pos > j:
                            continue
                        k_group = (j - first_pos) // d + 1
                        changes_group = 0
                        for x in range(k_group // 2):
                            left_pos = first_pos + x * d
                            right_pos = first_pos + (k_group - 1 - x) * d
                            if s[left_pos] != s[right_pos]:
                                changes_group += 1
                        total += changes_group
                    if total < min_cost:
                        min_cost = total
                cost[i][j] = min_cost
        
        # Initialize DP table
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for t in range(1, k + 1):
                # Iterate all possible j where the t-th partition starts at j
                for j in range(i - 1):
                    length = i - j
                    if length < 2:
                        continue
                    current_cost = cost[j][i - 1]
                    if current_cost == INF:
                        continue
                    if dp[j][t - 1] + current_cost < dp[i][t]:
                        dp[i][t] = dp[j][t - 1] + current_cost
        
        return dp[n][k]