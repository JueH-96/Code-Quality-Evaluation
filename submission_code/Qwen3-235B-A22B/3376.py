from typing import List

class TrieNode:
    __slots__ = ['children', 'best']
    def __init__(self):
        self.children = {}
        self.best = (float('inf'), -1)

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        
        for index, word in enumerate(wordsContainer):
            length = len(word)
            reversed_word = word[::-1]
            current_node = root
            
            # Update the root's best if necessary
            if length < current_node.best[0] or (length == current_node.best[0] and index < current_node.best[1]):
                current_node.best = (length, index)
            
            # Traverse each character in reversed_word
            for c in reversed_word:
                if c not in current_node.children:
                    # Create a new node with current word's best
                    current_node.children[c] = TrieNode()
                    current_node.children[c].best = (length, index)
                else:
                    # Update child's best if current word is better
                    child = current_node.children[c]
                    if (length < child.best[0]) or (length == child.best[0] and index < child.best[1]):
                        child.best = (length, index)
                # Move to the child node
                current_node = current_node.children[c]
        
        ans = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current_node = root
            for c in reversed_q:
                if c in current_node.children:
                    current_node = current_node.children[c]
                else:
                    break
            ans.append(current_node.best[1])
        
        return ans