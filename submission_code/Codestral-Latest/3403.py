class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(substring):
            char_count = {}
            for char in substring:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
            return len(set(char_count.values())) == 1

        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n]