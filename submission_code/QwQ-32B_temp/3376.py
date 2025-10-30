class Node:
    __slots__ = ['children', 'best_index', 'best_length']
    def __init__(self):
        self.children = {}
        self.best_index = -1
        self.best_length = float('inf')

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = Node()
        
        for idx, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current_length = len(word)
            current_node = root
            
            # Update root node first
            if (current_node.best_index == -1 or
                current_length < current_node.best_length or
                (current_length == current_node.best_length and idx < current_node.best_index)):
                current_node.best_index = idx
                current_node.best_length = current_length
            
            # Process each character in reversed_word
            for c in reversed_word:
                if c not in current_node.children:
                    current_node.children[c] = Node()
                current_node = current_node.children[c]
                
                # Update current node's best index and length
                if (current_node.best_index == -1 or
                    current_length < current_node.best_length or
                    (current_length == current_node.best_length and idx < current_node.best_index)):
                    current_node.best_index = idx
                    current_node.best_length = current_length
        
        ans = []
        for q in wordsQuery:
            reversed_q = q[::-1]
            current_node = root
            deepest_node = root  # Start with root as the default (empty suffix)
            
            for c in reversed_q:
                if c in current_node.children:
                    current_node = current_node.children[c]
                    deepest_node = current_node
                else:
                    break  # Can't go further, stop here
            
            ans.append(deepest_node.best_index)
        
        return ans