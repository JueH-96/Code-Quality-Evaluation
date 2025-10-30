class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Build a trie for suffix matching
        trie = {}
        for i, word in enumerate(wordsContainer):
            node = trie
            for char in word[::-1]:
                if char not in node:
                    node[char] = {}
                node = node[char]
            if '#' not in node or len(word) <= len(wordsContainer[node['#']]):
                node['#'] = i

        def find_longest_suffix(query):
            node = trie
            for char in query[::-1]:
                if char not in node:
                    break
                node = node[char]
            while '#' not in node and node:
                node = next(iter(node.values()))
            return node.get('#', 0)

        return [find_longest_suffix(query) for query in wordsQuery]