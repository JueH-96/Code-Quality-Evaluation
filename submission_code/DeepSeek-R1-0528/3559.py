class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        root = {}
        for word in words:
            node = root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        n = len(target)
        dp = [10**9] * (n + 1)
        dp[n] = 0
        
        for i in range(n - 1, -1, -1):
            node = root
            for j in range(i, n):
                char = target[j]
                if char not in node:
                    break
                node = node[char]
                if dp[j + 1] != 10**9:
                    dp[i] = min(dp[i], 1 + dp[j + 1])
        
        return dp[0] if dp[0] != 10**9 else -1