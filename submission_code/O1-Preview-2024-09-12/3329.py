class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        class TrieNode:
            def __init__(self):
                self.children = [None] * 10  # For digits '0'-'9'

        root = TrieNode()

        # Build Trie with numbers from arr1
        for num in arr1:
            node = root
            num_str = str(num)
            for ch in num_str:
                idx = int(ch)
                if not node.children[idx]:
                    node.children[idx] = TrieNode()
                node = node.children[idx]

        # For each number in arr2, traverse the Trie and find the longest common prefix length
        max_length = 0
        for num in arr2:
            node = root
            num_str = str(num)
            current_length = 0
            for ch in num_str:
                idx = int(ch)
                if node.children[idx]:
                    node = node.children[idx]
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    break  # No further match
        return max_length