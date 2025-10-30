class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Preprocess words to store all possible prefixes
        prefix_set = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefix_set.add(word[:i])
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, n + 1):
                current_prefix = target[i:j]
                if current_prefix in prefix_set:
                    if dp[j] > dp[i] + 1:
                        dp[j] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1