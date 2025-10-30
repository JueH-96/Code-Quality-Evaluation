class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Build a trie to store the XOR values of all subarrays
        root = {}
        for i in range(n):
            node = root
            for j in range(31, -1, -1):
                bit = (nums[i] >> j) & 1
                if bit not in node:
                    node[bit] = {}
                node = node[bit]
            node['#'] = nums[i]
        
        # Traverse the trie to find the maximum XOR score for each query
        def find_max_xor(l, r):
            node = root
            max_xor = 0
            for j in range(31, -1, -1):
                target_bit = 1 - ((nums[r] >> j) & 1)
                if target_bit in node:
                    max_xor |= (1 << j)
                    node = node[target_bit]
                else:
                    node = node[target_bit ^ 1]
            return max_xor ^ nums[r]
        
        return [find_max_xor(l, r) for l, r in queries]