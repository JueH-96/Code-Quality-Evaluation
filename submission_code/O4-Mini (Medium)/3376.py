from typing import List

class Solution:
    class TrieNode:
        __slots__ = ('children', 'best_index', 'best_length')
        def __init__(self):
            self.children = {}
            self.best_index = -1
            self.best_length = float('inf')

    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build a reversed trie of wordsContainer, storing at each node
        # the container index with the shortest length (and earliest index on ties).
        root = Solution.TrieNode()
        
        # Insert each word in reversed form into the trie
        for idx, word in enumerate(wordsContainer):
            length = len(word)
            node = root
            # Update root's best candidate
            if length < node.best_length or (length == node.best_length and idx < node.best_index):
                node.best_length = length
                node.best_index = idx
            # Traverse / create nodes for reversed word
            for ch in reversed(word):
                if ch not in node.children:
                    node.children[ch] = Solution.TrieNode()
                node = node.children[ch]
                # Update this node's best candidate
                if length < node.best_length or (length == node.best_length and idx < node.best_index):
                    node.best_length = length
                    node.best_index = idx
        
        # For each query, traverse as far as possible in the trie along reversed query.
        # The deepest node we reach stores the best container index for the longest suffix.
        ans = []
        for q in wordsQuery:
            node = root
            for ch in reversed(q):
                if ch in node.children:
                    node = node.children[ch]
                else:
                    break
            # At this node, node.best_index is our answer
            ans.append(node.best_index)
        
        return ans

# Example usage:
# sol = Solution()
# print(sol.stringIndices(["abcd","bcd","xbcd"], ["cd","bcd","xyz"]))  # [1, 1, 1]
# print(sol.stringIndices(["abcdefgh","poiuygh","ghghgh"], ["gh","acbfgh","acbfegh"]))  # [2, 0, 2]