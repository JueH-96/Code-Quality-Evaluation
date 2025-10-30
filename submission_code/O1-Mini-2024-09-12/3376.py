class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class TrieNode:
            __slots__ = ['children', 'best_idx']
            def __init__(self):
                self.children = [-1] * 26
                self.best_idx = -1

        # Precompute best index for empty suffix
        min_length = float('inf')
        best_empty_idx = -1
        for idx, word in enumerate(wordsContainer):
            if len(word) < min_length or (len(word) == min_length and idx < best_empty_idx):
                min_length = len(word)
                best_empty_idx = idx

        # Initialize trie
        nodes = [TrieNode()]
        
        def update_best(current_best, new_idx):
            if current_best == -1:
                return new_idx
            if len(wordsContainer[new_idx]) < len(wordsContainer[current_best]):
                return new_idx
            if len(wordsContainer[new_idx]) == len(wordsContainer[current_best]) and new_idx < current_best:
                return new_idx
            return current_best
        
        # Insert reversed words into trie
        for idx, word in enumerate(wordsContainer):
            current = 0
            nodes[current].best_idx = update_best(nodes[current].best_idx, idx)
            for ch in reversed(word):
                c = ord(ch) - ord('a')
                if nodes[current].children[c] == -1:
                    nodes.append(TrieNode())
                    nodes[current].children[c] = len(nodes) - 1
                current = nodes[current].children[c]
                nodes[current].best_idx = update_best(nodes[current].best_idx, idx)
        
        # Process queries
        ans = []
        for query in wordsQuery:
            current = 0
            best = nodes[current].best_idx
            for ch in reversed(query):
                c = ord(ch) - ord('a')
                if nodes[current].children[c] == -1:
                    break
                current = nodes[current].children[c]
                best = nodes[current].best_idx
            if best == -1:
                best = best_empty_idx
            ans.append(best)
        
        return ans