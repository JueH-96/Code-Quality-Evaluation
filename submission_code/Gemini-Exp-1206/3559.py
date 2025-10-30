class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes.add(word[:i])
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, n + 1):
                if target[i:j] in prefixes:
                    dp[j] = min(dp[j], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1