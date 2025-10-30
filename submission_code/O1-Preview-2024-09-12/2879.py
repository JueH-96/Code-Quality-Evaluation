class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        N = len(s)
        INF = float('inf')
        cost = [[INF] * (N+1) for _ in range(N+1)]  # cost[j][i]: cost to convert s[j:i] to semi-palindrome
        for j in range(N):
            for i in range(j+1, N+1):
                L = i - j
                min_cost = INF
                for d in range(1, L):
                    if L % d == 0:
                        total_cost = 0
                        for k_mod in range(d):
                            positions = []
                            for p in range(j, i):
                                if (p - j) % d == k_mod:
                                    positions.append(s[p])
                            m = len(positions)
                            c_list = positions
                            changes = 0
                            for l in range(m // 2):
                                if c_list[l] != c_list[m - l - 1]:
                                    changes += 1
                            total_cost += changes
                        min_cost = min(min_cost, total_cost)
                if min_cost == INF:
                    cost[j][i] = INF  # No valid d found
                else:
                    cost[j][i] = min_cost
        # Initialize dp table
        dp = [[INF] * (k+1) for _ in range(N+1)]  # dp[i][l]: min cost to partition s[0:i] into l substrings
        dp[0][0] = 0
        for i in range(1, N+1):
            dp[i][0] = INF
        # Fill dp table
        for l in range(1, k+1):
            for i in range(1, N+1):
                dp[i][l] = INF
                for j in range(l-1, i):
                    if dp[j][l-1] != INF and cost[j][i] != INF:
                        dp[i][l] = min(dp[i][l], dp[j][l-1] + cost[j][i])
        return dp[N][k] if dp[N][k] != INF else -1