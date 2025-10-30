class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert all numbers to strings
        str_arr1 = [str(x) for x in arr1]
        str_arr2 = [str(x) for x in arr2]

        # Build a Trie over arr1
        # Each node will have a dictionary of children keyed by digits
        # and no need for a terminal marker since we're only interested in
        # matching prefixes.
        class TrieNode:
            def __init__(self):
                self.children = {}

        root = TrieNode()

        # Insert each string from arr1 into the Trie
        for s in str_arr1:
            current = root
            for ch in s:
                if ch not in current.children:
                    current.children[ch] = TrieNode()
                current = current.children[ch]

        # Function to find the length of the common prefix
        # of a given string with the Trie
        def common_prefix_length(s: str) -> int:
            current = root
            length = 0
            for ch in s:
                if ch in current.children:
                    current = current.children[ch]
                    length += 1
                else:
                    break
            return length

        # For each string in str_arr2, compute the prefix match length
        # in the Trie and keep track of the maximum
        max_prefix_len = 0
        for s in str_arr2:
            max_prefix_len = max(max_prefix_len, common_prefix_length(s))

        return max_prefix_len