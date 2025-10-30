from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_xor = [0] * n

        # Compute the prefix XOR array
        prefix_xor[0] = nums[0]
        for i in range(1, n):
            prefix_xor[i] = prefix_xor[i-1] ^ nums[i]

        def get_xor_score(l, r):
            if l == 0:
                return prefix_xor[r]
            return prefix_xor[r] ^ prefix_xor[l-1]

        results = []
        for l, r in queries:
            max_xor_score = 0
            for start in range(l, r+1):
                for end in range(start, r+1):
                    max_xor_score = max(max_xor_score, get_xor_score(start, end))
            results.append(max_xor_score)

        return results