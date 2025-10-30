class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(substring):
            if not substring:
                return True
            counts = {}
            for char in substring:
                counts[char] = counts.get(char, 0) + 1
            if not counts:
                return True
            first_count = -1
            for count in counts.values():
                if first_count == -1:
                    first_count = count
                if count != first_count:
                    return False
            return True

        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0

        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                substring = s[i:j+1]
                if is_balanced(substring):
                    dp[i] = min(dp[i], 1 + dp[j+1])

        return dp[0]