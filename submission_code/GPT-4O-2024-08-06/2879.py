class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        def min_changes_to_semi_palindrome(sub):
            n = len(sub)
            min_changes = float('inf')
            for d in range(1, n):
                if n % d == 0:
                    changes = 0
                    for i in range(d):
                        freq = [0] * 26
                        count = 0
                        for j in range(i, n, d):
                            freq[ord(sub[j]) - ord('a')] += 1
                            count += 1
                        max_freq = max(freq)
                        changes += count - max_freq
                    min_changes = min(min_changes, changes)
            return min_changes if min_changes != float('inf') else 0

        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for l in range(i):
                    dp[i][j] = min(dp[i][j], dp[l][j - 1] + min_changes_to_semi_palindrome(s[l:i]))

        return dp[n][k]