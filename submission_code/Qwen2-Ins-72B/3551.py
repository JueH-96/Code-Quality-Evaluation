class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Precompute prefix XORs
        prefix_xor = [0] * (n + 1)
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        # Trie node class
        class TrieNode:
            def __init__(self):
                self.children = {}
        
        # Build the Trie
        root = TrieNode()
        for i, xor_val in enumerate(prefix_xor):
            node = root
            for j in range(31, -1, -1):
                bit = (xor_val >> j) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
        
        # Query function
        def query(l, r):
            xor_val = prefix_xor[r + 1] ^ prefix_xor[l]
            node = root
            max_xor = 0
            for j in range(31, -1, -1):
                bit = (xor_val >> j) & 1
                toggled_bit = 1 - bit
                if toggled_bit in node.children:
                    node = node.children[toggled_bit]
                    max_xor += (1 << j)
                else:
                    node = node.children[bit]
            return max_xor
        
        # Process queries
        return [query(l, r) for l, r in queries]