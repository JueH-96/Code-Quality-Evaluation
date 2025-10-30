class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.best_index = -1
        
        words_len = [len(word) for word in wordsContainer]
        n = len(wordsContainer)
        
        # Determine the global best index (shortest length, smallest index)
        global_best_index = 0
        min_length = words_len[0]
        for i in range(1, n):
            if words_len[i] < min_length:
                min_length = words_len[i]
                global_best_index = i
            elif words_len[i] == min_length:
                if i < global_best_index:
                    global_best_index = i
        
        # Initialize the trie with the root node
        root = TrieNode()
        root.best_index = global_best_index
        
        # Build the trie with reversed words from wordsContainer
        for idx in range(n):
            reversed_word = wordsContainer[idx][::-1]
            current = root
            for char in reversed_word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
                # Update the best index for the current node
                if current.best_index == -1:
                    current.best_index = idx
                else:
                    j = current.best_index
                    if words_len[idx] < words_len[j]:
                        current.best_index = idx
                    elif words_len[idx] == words_len[j] and idx < j:
                        current.best_index = idx
        
        # Process each query
        result = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current = root
            best_idx = root.best_index
            for char in reversed_q:
                if char in current.children:
                    current = current.children[char]
                    best_idx = current.best_index
                else:
                    break
            result.append(best_idx)
        
        return result