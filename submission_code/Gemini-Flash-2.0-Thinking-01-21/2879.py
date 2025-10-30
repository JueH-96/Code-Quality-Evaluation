class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        
        def get_semi_palindrome_cost(substring):
            sub_len = len(substring)
            if sub_len < 2:
                return 0
            min_cost = float('inf')
            for d in range(1, sub_len):
                if sub_len % d == 0:
                    current_cost = 0
                    for start_index in range(d):
                        group_str = ""
                        for index in range(start_index, sub_len, d):
                            group_str += substring[index]
                        palindrome_changes = 0
                        for p in range(len(group_str) // 2):
                            if group_str[p] != group_str[len(group_str) - 1 - p]:
                                palindrome_changes += 1
                        current_cost += palindrome_changes
                    min_cost = min(min_cost, current_cost)
            return min_cost

        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(j - 1, i):
                    cost = get_semi_palindrome_cost(s[l:i])
                    if dp[l][j - 1] != float('inf'):
                        dp[i][j] = min(dp[i][j], dp[l][j - 1] + cost)

        return dp[n][k]