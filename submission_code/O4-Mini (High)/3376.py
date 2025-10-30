from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # A trie node storing children and the best container word (by shortest length, then earliest index)
        class Node:
            __slots__ = ('children', 'best_len', 'best_idx')
            def __init__(self):
                self.children = {}
                self.best_len = float('inf')
                self.best_idx = -1

        # Build a trie of all reversed container words.
        # Each node keeps track of the "best" container word passing through it.
        root = Node()
        for idx, w in enumerate(wordsContainer):
            l = len(w)
            # Update root (which represents suffix of length 0)
            if l < root.best_len or (l == root.best_len and idx < root.best_idx):
                root.best_len = l
                root.best_idx = idx

            node = root
            # Insert reversed word into trie, updating best candidate at each node
            for ch in reversed(w):
                if ch not in node.children:
                    node.children[ch] = Node()
                node = node.children[ch]
                if l < node.best_len or (l == node.best_len and idx < node.best_idx):
                    node.best_len = l
                    node.best_idx = idx

        # Answer each query by walking down the trie along its reversed characters
        ans = []
        for q in wordsQuery:
            node = root
            best = node.best_idx
            for ch in reversed(q):
                if ch not in node.children:
                    break
                node = node.children[ch]
                best = node.best_idx
            ans.append(best)

        return ans