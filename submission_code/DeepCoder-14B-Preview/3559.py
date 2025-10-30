class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        valid_prefixes = set()
        max_word_length = 0
        
        for word in words:
            max_word_length = max(max_word_length, len(word))
            for i in range(1, len(word) + 1):
                valid_prefixes.add(word[:i])
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            max_j = min(max_word_length, i)
            for j in range(1, max_j + 1):
                start = i - j
                substring = target[start:i]
                if substring in valid_prefixes:
                    if dp[start] + 1 < dp[i]:
                        dp[i] = dp[start] + 1
        
        return dp[n] if dp[n] != float('inf') else -1