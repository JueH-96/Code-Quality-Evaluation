class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        dp = {}

        def solve(i, j, k):
            if i > j:
                return 0
            if i == j:
                return 1
            if (i, j, k) in dp:
                return dp[(i, j, k)]

            if abs(ord(s[i]) - ord(s[j])) <= k or 26 - abs(ord(s[i]) - ord(s[j])) <= k:
                dp[(i, j, k)] = 2 + solve(i + 1, j - 1, k - abs(ord(s[i]) - ord(s[j])))
                return dp[(i, j, k)]
            else:
                dp[(i, j, k)] = max(solve(i + 1, j, k), solve(i, j - 1, k))
                return dp[(i, j, k)]

        return solve(0, n - 1, k)