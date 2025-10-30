class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import defaultdict
        
        # Build a trie of all words (storing all prefixes).
        class TrieNode:
            def __init__(self):
                self.children = defaultdict(TrieNode)
        
        root = TrieNode()
        
        # Inserting each word into the trie:
        # For this problem, we only need to ensure
        # that each character path exists for prefixes.
        for w in words:
            node = root
            for char in w:
                node = node.children[char]
        
        n = len(target)
        # dp[i] = minimum number of valid strings to form target[:i]
        dp = [float('inf')] * (n+1)
        dp[0] = 0  # No strings needed to form empty prefix
        
        # For each position i in target, if dp[i] is not inf,
        # try to match a prefix of some word using the trie.
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = root
            # Attempt to extend from index i onwards
            j = i
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                j += 1
                dp[j] = min(dp[j], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1