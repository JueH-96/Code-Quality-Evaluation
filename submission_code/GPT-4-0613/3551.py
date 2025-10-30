class Solution:
    def maximumSubarrayXor(self, nums, queries):
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] ^ nums[i]
        trie = {}
        for i in range(n + 1):
            node = trie
            for j in range(31, -1, -1):
                bit = (prefix[i] >> j) & 1
                if bit not in node:
                    node[bit] = {}
                node = node[bit]
        res = []
        for l, r in queries:
            node = trie
            xor = prefix[r + 1] ^ prefix[l]
            for j in range(31, -1, -1):
                bit = (xor >> j) & 1
                if 1 - bit in node:
                    xor ^= (1 - bit) << j
                    node = node[1 - bit]
                else:
                    node = node[bit]
            res.append(xor)
        return res