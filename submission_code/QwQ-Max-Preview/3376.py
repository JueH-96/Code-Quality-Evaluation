class TrieNode:
    __slots__ = ['children', 'best_length', 'best_index']
    def __init__(self):
        self.children = {}
        self.best_length = float('inf')
        self.best_index = -1

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        
        for idx, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current = root
            word_len = len(word)
            
            # Update the root node's best if current word is better
            if word_len < current.best_length or (word_len == current.best_length and idx < current.best_index):
                current.best_length = word_len
                current.best_index = idx
            
            for c in reversed_word:
                if c not in current.children:
                    current.children[c] = TrieNode()
                current = current.children[c]
                # Update current node's best if current word is better
                if word_len < current.best_length or (word_len == current.best_length and idx < current.best_index):
                    current.best_length = word_len
                    current.best_index = idx
        
        ans = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current = root
            best_idx = current.best_index  # Start with root's best
            
            for c in reversed_q:
                if c in current.children:
                    current = current.children[c]
                    best_idx = current.best_index
                else:
                    break
            ans.append(best_idx)
        
        return ans