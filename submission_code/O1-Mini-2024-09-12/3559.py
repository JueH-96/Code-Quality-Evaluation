class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        class TrieNode:
            def __init__(self):
                self.children = {}
        
        root = TrieNode()
        
        # Build the trie with all prefixes of words
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
        
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 prefixes
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = root
            j = i
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                j += 1
                if dp[j] > dp[i] + 1:
                    dp[j] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1