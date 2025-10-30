class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.best_idx = -1
                self.best_len = float('inf')
        
        # Build Trie with reversed words
        root = TrieNode()
        
        for idx, word in enumerate(wordsContainer):
            node = root
            # Update root (empty suffix case)
            if len(word) < node.best_len:
                node.best_idx = idx
                node.best_len = len(word)
            
            # Insert word in reverse order
            for i in range(len(word) - 1, -1, -1):
                char = word[i]
                if char not in node.children:
                    node.children[char] = TrieNode()
                
                node = node.children[char]
                
                # Update best index for this suffix
                if len(word) < node.best_len:
                    node.best_idx = idx
                    node.best_len = len(word)
        
        result = []
        for query in wordsQuery:
            # Find longest matching suffix
            node = root
            
            for i in range(len(query) - 1, -1, -1):
                char = query[i]
                if char in node.children:
                    node = node.children[char]
                else:
                    break
            
            result.append(node.best_idx)
        
        return result