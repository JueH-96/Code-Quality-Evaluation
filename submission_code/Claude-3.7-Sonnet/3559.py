class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        # dp[i] represents the minimum number of valid strings needed to form target[0...i-1]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Precompute all valid prefixes
        valid_prefixes = set()
        max_prefix_length = 0
        for word in words:
            for i in range(1, len(word) + 1):
                valid_prefixes.add(word[:i])
                max_prefix_length = max(max_prefix_length, i)
        
        for i in range(1, n + 1):
            for j in range(max(0, i-max_prefix_length), i):
                if dp[j] != float('inf') and target[j:i] in valid_prefixes:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1