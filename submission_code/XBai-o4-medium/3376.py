class TrieNode:
    __slots__ = ['children', 'best_index', 'best_length']
    def __init__(self):
        self.children = dict()
        self.best_index = -1
        self.best_length = float('inf')

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        for idx, word in enumerate(wordsContainer):
            current = root
            # Update root's best if current word is better
            if (current.best_length > len(word)) or (current.best_length == len(word) and idx < current.best_index):
                current.best_index = idx
                current.best_length = len(word)
            # Process reversed word
            reversed_word = word[::-1]
            for c in reversed_word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
                # Update current node's best if needed
                if (current.best_length > len(word)) or (current.best_length == len(word) and idx < current.best_index):
                    current.best_index = idx
                    current.best_length = len(word)
        
        ans = []
        for q in wordsQuery:
            current = root
            reversed_q = q[::-1]
            for c in reversed_q:
                if c in current.children:
                    current = current.children[c]
                else:
                    break
            ans.append(current.best_index)
        return ans