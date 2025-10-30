class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        class TrieNode:
            def __init__(self):
                self.children = {}
        root = TrieNode()
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if dp[i] != float('inf'):
                node = root
                pos = i
                while pos < n and target[pos] in node.children:
                    node = node.children[target[pos]]
                    pos += 1
                    dp[pos] = min(dp[pos], dp[i] + 1)
        return dp[n] if dp[n] != float('inf') else -1