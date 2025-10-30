class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)

        # Helper function to check if a string is a semi-palindrome
        def is_semi_palindrome(sub):
            length = len(sub)
            for d in range(1, length):
                if length % d == 0:
                    if all(sub[i] == sub[i % d] for i in range(length)):
                        return True
            return False

        # Helper function to calculate the minimum changes needed to make a substring a semi-palindrome
        def min_changes_to_semi_palindrome(sub):
            length = len(sub)
            min_changes = float('inf')
            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    for i in range(length):
                        if sub[i] != sub[i % d]:
                            changes += 1
                    min_changes = min(min_changes, changes)
            return min_changes

        # Dynamic Programming table to store the minimum changes
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for l in range(1, i + 1):
                    sub = s[i - l:i]
                    if is_semi_palindrome(sub):
                        dp[i][j] = min(dp[i][j], dp[i - l][j - 1])
                    else:
                        dp[i][j] = min(dp[i][j], dp[i - l][j - 1] + min_changes_to_semi_palindrome(sub))

        return dp[n][k]