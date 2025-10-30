class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # prefix_counts[i][c] = count of char c in s[:i]
        prefix_counts = [[0] * 26 for _ in range(n + 1)]
        for i, ch in enumerate(s, 1):
            ci = ord(ch) - ord('a')
            for c in range(26):
                prefix_counts[i][c] = prefix_counts[i - 1][c]
            prefix_counts[i][ci] += 1

        # dp[i] = minimum # of balanced substrings to cover s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            # try every possible cut ending at i
            for j in range(i):
                # counts of each char in s[j:i]
                counts = [
                    prefix_counts[i][c] - prefix_counts[j][c]
                    for c in range(26)
                ]
                # collect non-zero counts
                nonzero = [cnt for cnt in counts if cnt > 0]
                # a substring is balanced if all non-zero counts are equal
                if nonzero and len(set(nonzero)) == 1:
                    dp[i] = min(dp[i], dp[j] + 1)
            # Note: single character substrings are balanced by definition
            # but the above logic already handles them (count = [1]).
        return dp[n]

# Example usage:
# sol = Solution()
# print(sol.minimumSubstringsInPartition("fabccddg"))   # Output: 3
# print(sol.minimumSubstringsInPartition("abababaccddb"))  # Output: 2