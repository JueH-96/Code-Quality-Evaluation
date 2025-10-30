class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        max_len = 200
        divisors_dict = {}
        for num in range(2, max_len + 1):
            divisors_dict[num] = [d for d in range(1, num) if num % d == 0]
        
        cost_table = [[float('inf')] * (n + 1) for _ in range(n)]
        
        for l in range(n):
            for r in range(l + 2, n + 1):
                L = r - l
                divisors = divisors_dict.get(L, [])
                best_cost = float('inf')
                for d in divisors:
                    total_changes = 0
                    for residue in range(d):
                        start_index = l + residue
                        m = L // d
                        low = 0
                        high = m - 1
                        changes_in_group = 0
                        while low < high:
                            if s[start_index + low * d] != s[start_index + high * d]:
                                changes_in_group += 1
                            low += 1
                            high -= 1
                        total_changes += changes_in_group
                        if total_changes >= best_cost:
                            break
                    if total_changes < best_cost:
                        best_cost = total_changes
                    if best_cost == 0:
                        break
                cost_table[l][r] = best_cost
        
        INF = 10**9
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for j in range(1, k + 1):
            for i in range(2 * j, n + 1):
                min_val = INF
                for p in range(2 * (j - 1), i - 1):
                    if dp[p][j - 1] + cost_table[p][i] < min_val:
                        min_val = dp[p][j - 1] + cost_table[p][i]
                dp[i][j] = min_val
        
        return dp[n][k]