class TrieNode:
    __slots__ = ['children']
    def __init__(self):
        self.children = {}

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        root = TrieNode()
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
        
        n = len(target)
        BIG = 10**9
        dp = [BIG] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == BIG:
                continue
            node = root
            j = i
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                j += 1
                if dp[i] + 1 < dp[j]:
                    dp[j] = dp[i] + 1
        
        return dp[n] if dp[n] != BIG else -1