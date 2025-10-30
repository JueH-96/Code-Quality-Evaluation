class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        divisors = {}
        for L in range(2, n + 1):
            divisors[L] = [d for d in range(1, L) if L % d == 0]
        
        min_cost = [[10**9] * n for _ in range(n)]
        
        for L in range(2, n + 1):
            for start in range(0, n - L + 1):
                end = start + L - 1
                current_min = 10**9
                for d in divisors[L]:
                    total = 0
                    t = L // d
                    for r in range(d):
                        for m in range(t // 2):
                            left_index = start + r + m * d
                            right_index = start + r + (t - 1 - m) * d
                            if s[left_index] != s[right_index]:
                                total += 1
                    current_min = min(current_min, total)
                min_cost[start][end] = current_min
        
        dp = [[10**9] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for k_count in range(1, k + 1):
            for i in range(2 * k_count, n + 1):
                for j in range((k_count - 1) * 2, i - 1):
                    prev_cost = dp[j][k_count - 1]
                    if prev_cost != 10**9:
                        cost_to_add = min_cost[j][i - 1]
                        new_cost = prev_cost + cost_to_add
                        if new_cost < dp[i][k_count]:
                            dp[i][k_count] = new_cost
        
        return dp[n][k]