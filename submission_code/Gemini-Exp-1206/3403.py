class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        def is_balanced(sub):
            if not sub:
                return False
            counts = {}
            for char in sub:
                counts[char] = counts.get(char, 0) + 1
            first_count = None
            for count in counts.values():
                if first_count is None:
                    first_count = count
                elif count != first_count:
                    return False
            return True

        for i in range(1, n + 1):
            for j in range(i):
                if is_balanced(s[j:i]):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n]