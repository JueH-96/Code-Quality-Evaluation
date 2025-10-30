from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_xor = [0] * (n + 1)
        
        # Compute prefix XOR array
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        answer = []
        
        for l, r in queries:
            max_xor = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    subarray_xor = prefix_xor[j + 1] ^ prefix_xor[i]
                    max_xor = max(max_xor, subarray_xor)
            answer.append(max_xor)
        
        return answer