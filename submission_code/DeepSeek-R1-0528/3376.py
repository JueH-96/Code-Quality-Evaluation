class TrieNode:
    __slots__ = ['children', 'best_index', 'best_length']
    def __init__(self):
        self.children = {}
        self.best_index = None
        self.best_length = float('inf')

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        
        for idx, word in enumerate(wordsContainer):
            rword = word[::-1]
            l = len(word)
            current = root
            if l < current.best_length or (l == current.best_length and idx < current.best_index):
                current.best_length = l
                current.best_index = idx
            for char in rword:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
                if l < current.best_length or (l == current.best_length and idx < current.best_index):
                    current.best_length = l
                    current.best_index = idx
        
        ans = []
        for word in wordsQuery:
            rword = word[::-1]
            current = root
            best_index = current.best_index
            for char in rword:
                if char in current.children:
                    current = current.children[char]
                    best_index = current.best_index
                else:
                    break
            ans.append(best_index)
        return ans