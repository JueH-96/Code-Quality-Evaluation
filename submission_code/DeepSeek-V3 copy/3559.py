class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Preprocess words to store all possible prefixes
        prefix_set = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefix_set.add(word[:i])
        
        # Dynamic programming array to store the minimum number of valid strings needed
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        for i in range(len(target)):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, len(target) + 1):
                substring = target[i:j]
                if substring in prefix_set:
                    if dp[j] > dp[i] + 1:
                        dp[j] = dp[i] + 1
        
        return dp[len(target)] if dp[len(target)] != float('inf') else -1