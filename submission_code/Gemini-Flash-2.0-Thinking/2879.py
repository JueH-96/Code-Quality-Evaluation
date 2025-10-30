class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        cost_memo = {}

        def get_semi_palindrome_cost(sub):
            if sub in cost_memo:
                return cost_memo[sub]

            m = len(sub)
            if m == 0:
                return 0

            min_changes = float('inf')
            for d in range(1, m):
                if m % d == 0:
                    total_changes = 0
                    for start in range(d):
                        subsequence = [sub[start + i * d] for i in range(m // d)]
                        len_subsequence = len(subsequence)
                        changes_sub = 0
                        for j in range(len_subsequence // 2):
                            if subsequence[j] != subsequence[len_subsequence - 1 - j]:
                                changes_sub += 1
                        total_changes += changes_sub
                    min_changes = min(min_changes, total_changes)

            cost_memo[sub] = min_changes if min_changes != float('inf') else float('inf')
            return cost_memo[sub]

        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for num_partitions in range(1, k + 1):
            for i in range(1, n + 1):
                for p in range(i):
                    cost_substring = get_semi_palindrome_cost(s[p:i])
                    if dp[p][num_partitions - 1] != float('inf') and cost_substring != float('inf'):
                        dp[i][num_partitions] = min(dp[i][num_partitions], dp[p][num_partitions - 1] + cost_substring)

        return dp[n][k] if dp[n][k] != float('inf') else 0