class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)

        # Helper function to calculate the minimum changes needed to make a substring a semi-palindrome
        def min_changes_to_semi_palindrome(substr):
            length = len(substr)
            min_changes = float('inf')

            for d in range(1, length):
                if length % d == 0:
                    changes = 0
                    for i in range(d):
                        left, right = i, length - i - 1
                        while left <= right:
                            if substr[left] != substr[right]:
                                changes += 1
                            left += d
                            right -= d
                    min_changes = min(min_changes, changes)

            return min_changes

        # Dynamic programming table to store the minimum changes
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for p in range(j, i + 1):
                    dp[i][j] = min(dp[i][j], dp[i - p][j - 1] + min_changes_to_semi_palindrome(s[i - p:i]))

        return dp[n][k]

# Example usage:
sol = Solution()
print(sol.minimumChanges("abcac", 2))  # Output: 1
print(sol.minimumChanges("abcdef", 2))  # Output: 2
print(sol.minimumChanges("aabbaa", 3))  # Output: 0