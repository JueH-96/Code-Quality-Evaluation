from typing import List

class TrieNode:
    __slots__ = ("children", "best_idx", "best_len")
    def __init__(self):
        self.children = {}
        # among all words passing through this node, we store:
        self.best_idx = None  # index in wordsContainer
        self.best_len = None  # length of that word

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build a reversed-trie of wordsContainer. At each node store the "best" word index
        # according to: longest common suffix (i.e. longest reversed-prefix), tie break on shorter word,
        # then on earlier index.
        root = TrieNode()
        
        # Insert each container word into trie (in reversed order)
        for idx, w in enumerate(wordsContainer):
            L = len(w)
            # First, update root (empty suffix)
            if root.best_idx is None or L < root.best_len or (L == root.best_len and idx < root.best_idx):
                root.best_idx = idx
                root.best_len = L
            
            node = root
            # traverse reversed word
            for ch in reversed(w):
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                # update this node's best if this word is a better candidate
                if node.best_idx is None or L < node.best_len or (L == node.best_len and idx < node.best_idx):
                    node.best_idx = idx
                    node.best_len = L
        
        # Now answer queries
        ans = []
        for q in wordsQuery:
            node = root
            best = node.best_idx  # best for empty suffix
            # traverse as far as match allows
            for ch in reversed(q):
                if ch in node.children:
                    node = node.children[ch]
                    best = node.best_idx
                else:
                    break
            ans.append(best)
        
        return ans