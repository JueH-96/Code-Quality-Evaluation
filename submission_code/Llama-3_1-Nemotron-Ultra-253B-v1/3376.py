from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class TrieNode:
            __slots__ = ['children', 'min_length', 'index']
            def __init__(self):
                self.children = {}
                self.min_length = float('inf')
                self.index = -1
        
        root = TrieNode()
        for idx, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current = root
            # Update root node
            if len(word) < current.min_length:
                current.min_length = len(word)
                current.index = idx
            elif len(word) == current.min_length and idx < current.index:
                current.index = idx
            # Process each character in reversed_word
            for c in reversed_word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
                # Update current node's min_length and index
                if len(word) < current.min_length:
                    current.min_length = len(word)
                    current.index = idx
                elif len(word) == current.min_length and idx < current.index:
                    current.index = idx
        
        ans = []
        for query in wordsQuery:
            reversed_query = query[::-1]
            current = root
            max_depth_node = root
            for c in reversed_query:
                if c in current.children:
                    current = current.children[c]
                    max_depth_node = current
                else:
                    break
            ans.append(max_depth_node.index)
        
        return ans