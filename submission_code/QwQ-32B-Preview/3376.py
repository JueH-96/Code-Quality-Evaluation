class TrieNode:
    def __init__(self):
        self.children = {}
        self.best_index = None  # Tuple of (length, index)

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        root = TrieNode()
        shortest_index = None

        # Function to insert reversed word into Trie
        def insert(word, index):
            nonlocal shortest_index
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                if node.best_index is None or node.best_index > (len(word), index):
                    node.best_index = (len(word), index)
            if shortest_index is None or shortest_index > (len(word), index):
                shortest_index = (len(word), index)

        # Insert all reversed words into Trie
        for idx, word in enumerate(wordsContainer):
            insert(word[::-1], idx)

        # Function to search for reversed query in Trie
        def search(query):
            node = root
            path = []
            for char in query:
                if char in node.children:
                    node = node.children[char]
                    path.append(node)
                else:
                    break
            # Find the best index from the deepest match upwards
            while path:
                candidate = path.pop().best_index
                if candidate is not None:
                    return candidate
            return root.best_index

        # Prepare the result
        result = []
        reversed_query = [q[::-1] for q in wordsQuery]
        for q in reversed_query:
            length, idx = search(q)
            result.append(idx)
        return result