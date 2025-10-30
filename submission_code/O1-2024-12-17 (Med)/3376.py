class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        from typing import List
        
        class TrieNode:
            __slots__ = ['children', 'best_index', 'best_length']
            def __init__(self):
                # Each node holds references for 26 letters and stores the best word info
                self.children = [None] * 26
                self.best_index = -1
                self.best_length = float('inf')
        
        def update_best(node: TrieNode, word_len: int, word_idx: int) -> None:
            """
            Updates the node's best word info based on:
            1) The minimum length.
            2) If the same length, by earliest index.
            """
            if word_len < node.best_length:
                node.best_length = word_len
                node.best_index = word_idx
            elif word_len == node.best_length:
                if word_idx < node.best_index:
                    node.best_index = word_idx
        
        # Build a trie of reversed words from wordsContainer
        root = TrieNode()
        for i, word in enumerate(wordsContainer):
            w_len = len(word)
            # Update the root for the empty suffix case
            update_best(root, w_len, i)
            current = root
            # Insert characters in reverse
            for ch in reversed(word):
                idx = ord(ch) - ord('a')
                if current.children[idx] is None:
                    current.children[idx] = TrieNode()
                current = current.children[idx]
                # Update the best word info for this node
                update_best(current, w_len, i)
        
        # For each query, traverse in reverse to get the best match
        ans = []
        for query in wordsQuery:
            best_idx = root.best_index  # start with whatever is stored at the root (empty suffix)
            current = root
            for ch in reversed(query):
                idx = ord(ch) - ord('a')
                # If we cannot proceed further, break early
                if current.children[idx] is None:
                    break
                current = current.children[idx]
                # Update the best index with node's best
                best_idx = current.best_index
            ans.append(best_idx)
        
        return ans