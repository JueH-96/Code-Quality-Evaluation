class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Create a suffix trie for efficient longest common suffix search
        trie = {}
        
        # For the root node, find the shortest string's index (for empty suffix)
        best_empty_idx = 0
        for idx, word in enumerate(wordsContainer):
            if len(word) < len(wordsContainer[best_empty_idx]) or (len(word) == len(wordsContainer[best_empty_idx]) and idx < best_empty_idx):
                best_empty_idx = idx
        
        # Initialize root with best empty suffix index
        trie["best_idx"] = best_empty_idx
        
        # Build the suffix trie by inserting words in reverse
        for idx, word in enumerate(wordsContainer):
            node = trie
            for char in reversed(word):
                if char not in node:
                    node[char] = {"best_idx": idx}
                node = node[char]
                
                # Update the best index at this node if needed
                curr_best_idx = node["best_idx"]
                if (len(word) < len(wordsContainer[curr_best_idx]) or 
                   (len(word) == len(wordsContainer[curr_best_idx]) and idx < curr_best_idx)):
                    node["best_idx"] = idx
        
        # Process each query
        result = []
        for query in wordsQuery:
            node = trie
            for char in reversed(query):
                if char in node:
                    node = node[char]
                else:
                    break
            
            # Return the best index at current node
            result.append(node["best_idx"])
        
        return result