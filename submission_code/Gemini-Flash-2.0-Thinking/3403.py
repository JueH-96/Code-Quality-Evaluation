class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        def is_balanced(sub):
            if not sub:
                return True
            counts = {}
            for char in sub:
                counts[char] = counts.get(char, 0) + 1
            if not counts:
                return True
            count = list(counts.values())[0]
            return all(c == count for c in counts.values())

        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if is_balanced(substring):
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n]