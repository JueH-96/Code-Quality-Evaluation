class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for word in words:
                m = len(word)
                if i >= m and target[i - m:i] == word:
                    dp[i] = min(dp[i], dp[i - m] + 1)

        return dp[n] if dp[n] != float('inf') else -1