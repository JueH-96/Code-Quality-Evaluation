class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        from functools import reduce
        from operator import xor
        
        def max_xor_subarray(nums):
            n = len(nums)
            max_xor = 0
            prefix_xor = 0
            trie = {}
            
            for num in nums:
                current = trie
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    if 1 - bit not in current:
                        current[1 - bit] = {}
                    current = current[1 - bit]
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    if bit in current:
                        prefix_xor ^= (1 << i)
                        current = current[bit]
                    else:
                        current = current[1 - bit]
                max_xor = max(max_xor, prefix_xor)
                prefix_xor ^= num
                current = trie
            
            return max_xor
        
        answer = []
        for l, r in queries:
            answer.append(max_xor_subarray(nums[l:r+1]))
        
        return answer