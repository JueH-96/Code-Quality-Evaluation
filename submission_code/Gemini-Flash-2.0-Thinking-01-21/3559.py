class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        valid_prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                valid_prefixes.add(word[:i])
        valid_prefixes = list(valid_prefixes)
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            for prefix in valid_prefixes:
                prefix_len = len(prefix)
                if i + prefix_len <= n and target[i:i+prefix_len] == prefix:
                    dp[i] = min(dp[i], 1 + dp[i + prefix_len])
        if dp[0] == float('inf'):
            return -1
        else:
            return dp[0]