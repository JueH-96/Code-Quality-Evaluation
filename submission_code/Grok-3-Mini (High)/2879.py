class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9
        # Precompute the min cost to make each substring a semi-palindrome
        cost_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                if length == 1:
                    cost_matrix[start][end] = INF
                else:
                    min_cost_d = INF
                    for d in range(1, length):
                        if length % d == 0:
                            cost_d = 0
                            m = length // d
                            for r in range(d):
                                pal_cost = 0
                                for k in range(m // 2):
                                    pos_left = start + (r + k * d)
                                    pos_right = start + (r + (m - 1 - k) * d)
                                    if s[pos_left] != s[pos_right]:
                                        pal_cost += 1
                                cost_d += pal_cost
                            if cost_d < min_cost_d:
                                min_cost_d = cost_d
                    cost_matrix[start][end] = min_cost_d
        # Now DP to partition into k substrings
        dp = [[INF for _ in range(k + 1)] for _ in range(n + 1)]
        dp[0][0] = 0
        for p in range(1, k + 1):
            for i in range(1, n + 1):
                min_val = INF
                for c in range(0, i):
                    sub_cost = cost_matrix[c][i - 1]
                    curr_cost = dp[c][p - 1] + sub_cost
                    if curr_cost < min_val:
                        min_val = curr_cost
                dp[i][p] = min_val
        return dp[n][k]