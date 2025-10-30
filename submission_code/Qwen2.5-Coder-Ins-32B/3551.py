from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_xor = [0] * (n + 1)
        
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        def xor_range(l: int, r: int) -> int:
            return prefix_xor[r + 1] ^ prefix_xor[l]
        
        answer = []
        for l, r in queries:
            max_xor = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    max_xor = max(max_xor, xor_range(i, j))
            answer.append(max_xor)
        
        return answer