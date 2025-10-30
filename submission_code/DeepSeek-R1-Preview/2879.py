class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        if n < 2 * k:
            return 0  # Impossible case, though constraints ensure this doesn't happen
        
        # Precompute the cost matrix
        INF = float('inf')
        cost = [[INF for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                len_sub = j - i + 1
                possible_ds = []
                for d in range(1, len_sub):
                    if len_sub % d == 0:
                        possible_ds.append(d)
                if not possible_ds:
                    continue
                min_cost = INF
                for d in possible_ds:
                    total_changes = 0
                    for r in range(d):
                        group = []
                        for pos in range(r, len_sub, d):
                            group.append(s[i + pos])
                        m = len(group)
                        changes = 0
                        for x in range(m // 2):
                            if group[x] != group[m - 1 - x]:
                                changes += 1
                        total_changes += changes
                    if total_changes < min_cost:
                        min_cost = total_changes
                cost[i][j] = min_cost
        
        # Initialize DP table
        dp = [[INF] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        
        for m in range(1, k + 1):
            for i in range(1, n + 1):
                for j in range(0, i):
                    if i - j >= 2 and j >= 2 * (m - 1):
                        if dp[m - 1][j] != INF:
                            current_cost = dp[m - 1][j] + cost[j][i - 1]
                            if current_cost < dp[m][i]:
                                dp[m][i] = current_cost
        
        return dp[k][n]