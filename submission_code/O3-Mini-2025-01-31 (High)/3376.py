from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        # candidate is a tuple (word_length, word_index) that represents the best container word
        # among those that pass through this node.
        self.cand = None

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build a Trie over the reversed container words.
        root = TrieNode()
        
        for i, word in enumerate(wordsContainer):
            candidate = (len(word), i)
            # Update candidate at the root, since every word passes through the root.
            if root.cand is None or candidate < root.cand:
                root.cand = candidate
            node = root
            # Insert the reversed word character by character.
            for ch in word[::-1]:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
                if node.cand is None or candidate < node.cand:
                    node.cand = candidate
        
        results = []
        # For each query, traverse the trie using the reversed query.
        for query in wordsQuery:
            node = root
            best_candidate = root.cand  # At least, common suffix "" matches all.
            for ch in query[::-1]:
                if ch in node.children:
                    node = node.children[ch]
                    best_candidate = node.cand
                else:
                    # No further match; break. The current best_candidate corresponds to the
                    # longest common suffix (i.e. prefix in reversed strings) we could get.
                    break
            results.append(best_candidate[1])
        
        return results