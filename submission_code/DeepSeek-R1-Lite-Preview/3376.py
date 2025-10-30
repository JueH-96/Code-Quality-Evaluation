class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Define the TrieNode class
        class TrieNode:
            def __init__(self):
                self.children = [None] * 26
                self.best_word = None  # Tuple of (length, index)
        
        # Function to insert a reversed word into the trie
        def insert(root, word, index):
            node = root
            length = len(word)
            for ch in word:
                idx = ord(ch) - ord('a')
                if node.children[idx] is None:
                    node.children[idx] = TrieNode()
                node = node.children[idx]
                # Update best_word if current word is better
                if node.best_word is None or (length < node.best_word[0]) or (length == node.best_word[0] and index < node.best_word[1]):
                    node.best_word = (length, index)
        
        # Build the trie with reversed words from wordsContainer
        root = TrieNode()
        for index, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            insert(root, reversed_word, index)
        
        # Function to query the trie with a reversed query string
        def query(root, query_str):
            node = root
            for ch in query_str:
                idx = ord(ch) - ord('a')
                if node.children[idx] is not None:
                    node = node.children[idx]
                else:
                    break
            # Return the best_word from the deepest matching node
            return node.best_word[1]
        
        # Prepare the answer list
        ans = []
        for q in wordsQuery:
            reversed_q = q[::-1]
            ans.append(query(root, reversed_q))
        
        return ans