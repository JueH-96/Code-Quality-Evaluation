class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build a Trie with all words
        trie = {}
        for word in words:
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            
            # Find all valid prefixes starting from position i
            node = trie
            for j in range(i, n):
                if target[j] not in node:
                    break
                node = node[target[j]]
                dp[j + 1] = min(dp[j + 1], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1