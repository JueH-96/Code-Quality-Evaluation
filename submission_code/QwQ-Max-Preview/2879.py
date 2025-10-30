class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        if n < 2:
            return 0  # Not possible per problem constraints, but handle edge case
        
        # Precompute the cost matrix
        cost = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                L = j - i + 1
                possible_ds = []
                for d in range(1, L):
                    if L % d == 0:
                        possible_ds.append(d)
                min_total = float('inf')
                for d in possible_ds:
                    m = L // d
                    total = 0
                    for r in range(d):
                        group = []
                        for k_idx in range(m):
                            pos = i + r + k_idx * d
                            group.append(s[pos])
                        group_cost = 0
                        for kk in range(len(group) // 2):
                            if group[kk] != group[-kk - 1]:
                                group_cost += 1
                        total += group_cost
                    if total < min_total:
                        min_total = total
                cost[i][j] = min_total
        
        # Initialize DP table
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for t in range(1, k + 1):
            for i in range(1, n + 1):
                # The current part is s[j..i-1], which must have length >=2
                for j in range(0, i - 1):
                    if (i - 1 - j + 1) >= 2:
                        if dp[j][t - 1] + cost[j][i - 1] < dp[i][t]:
                            dp[i][t] = dp[j][t - 1] + cost[j][i - 1]
        
        return dp[n][k] if dp[n][k] != float('inf') else -1