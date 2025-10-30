class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        valid_prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                valid_prefixes.add(word[:i])

        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                substring = target[j:i]
                if substring in valid_prefixes:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1