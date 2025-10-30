from typing import List

class Solution:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.best = None  # (index, length)

    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = self.TrieNode()
        # Insert all words into the trie
        for idx, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current = root
            # Update the root's best candidate
            if current.best is None:
                current.best = (idx, len(word))
            else:
                if (len(word) < current.best[1]) or (len(word) == current.best[1] and idx < current.best[0]):
                    current.best = (idx, len(word))
            # Process each character in the reversed word
            for c in reversed_word:
                if c not in current.children:
                    current.children[c] = self.TrieNode()
                current = current.children[c]
                # Update the current node's best candidate
                if current.best is None or (len(word) < current.best[1]) or (len(word) == current.best[1] and idx < current.best[0]):
                    current.best = (idx, len(word))
        
        # Process each query
        ans = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current = root
            max_node = root
            for c in reversed_q:
                if c in current.children:
                    current = current.children[c]
                    max_node = current
                else:
                    break
            ans.append(max_node.best[0] if max_node.best else 0)
        return ans