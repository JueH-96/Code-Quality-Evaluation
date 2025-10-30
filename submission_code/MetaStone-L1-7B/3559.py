class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        if n == 0:
            return 0
        
        match = [False] * (n + 1)
        for word in words:
            max_l = 0
            word_len = len(word)
            for l in range(1, min(word_len, n) + 1):
                if word[:l] == target[:l]:
                    max_l = l
            for l in range(1, max_l + 1):
                match[l] = True
        
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for l in range(1, i + 1):
                if match[l]:
                    if dp[i - l] + 1 < dp[i]:
                        dp[i] = dp[i - l] + 1
        
        return dp[n] if dp[n] != INF else -1