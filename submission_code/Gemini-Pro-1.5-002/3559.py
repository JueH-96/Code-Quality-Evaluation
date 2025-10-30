class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes.add(word[:i])

        for i in range(1, n + 1):
            for j in range(1, i + 1):
                if target[i-j:i] in prefixes:
                    dp[i] = min(dp[i], dp[i-j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1