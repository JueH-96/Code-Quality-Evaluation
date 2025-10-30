from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        node = self.root
        # insert all characters; note that every prefix is implicitly valid
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build Trie from all words. Since any prefix is valid, we don't need to mark endpoints.
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        n = len(target)
        # dp[i] = minimum number of segments to form target[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # For each starting index, try to match as many characters as possible.
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            node = trie.root
            j = i
            # Try to extend the matching prefix from target[i:]
            while j < n and target[j] in node.children:
                node = node.children[target[j]]
                j += 1
                # Since any prefix matched along the way is valid,
                # update dp[j]
                if dp[j] > dp[i] + 1:
                    dp[j] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1