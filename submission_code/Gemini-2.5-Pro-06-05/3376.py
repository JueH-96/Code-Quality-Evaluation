class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        """
        Finds the index of the best matching string in wordsContainer for each query.
        The solution uses a Trie built on reversed strings to efficiently find the
        longest common suffix.
        """

        # A nested class for the Trie node to keep the implementation self-contained.
        class TrieNode:
            def __init__(self):
                # children maps a character to a child TrieNode
                self.children = {}
                # min_len stores the length of the shortest word passing through this node
                self.min_len = float('inf')
                # best_index stores the index of that shortest word
                self.best_index = -1

        root = TrieNode()
        
        # Build the Suffix Trie.
        # We iterate through wordsContainer to populate the trie. The trie is built
        # on reversed words to facilitate suffix matching.
        for i, word in enumerate(wordsContainer):
            curr = root
            word_len = len(word)
            
            # Every word is a candidate for the empty suffix, represented by the root.
            # We update the root's info if this word is a better candidate (shorter).
            # The tie-breaking rule (smallest index for same length) is handled
            # implicitly by processing words in order and only updating on strictly
            # smaller lengths.
            if word_len < curr.min_len:
                curr.min_len = word_len
                curr.best_index = i
            
            # Traverse the trie for each character of the reversed word, updating
            # node information along the path.
            for char in reversed(word):
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                
                curr = curr.children[char]
                
                # Update the node if the current word is a better candidate for the
                # suffix represented by this node.
                if word_len < curr.min_len:
                    curr.min_len = word_len
                    curr.best_index = i
        
        # Process each query.
        ans = []
        for query in wordsQuery:
            curr = root
            # Traverse the trie using the reversed query to find the longest common suffix.
            for char in reversed(query):
                if char in curr.children:
                    # Move to the next node if the suffix character matches.
                    curr = curr.children[char]
                else:
                    # Stop if there's no path for the current character.
                    # `curr` now corresponds to the node for the longest found common suffix.
                    break
            
            # The `best_index` stored in the final node is the answer for this query.
            ans.append(curr.best_index)
            
        return ans