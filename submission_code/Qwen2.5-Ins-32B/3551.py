from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_xor = [0] * (n + 1)
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        def query(l, r):
            subarray_xor = prefix_xor[r + 1] ^ prefix_xor[l]
            max_score = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    score = prefix_xor[j + 1] ^ prefix_xor[i]
                    max_score = max(max_score, score)
            return max_score
        
        answer = []
        for l, r in queries:
            answer.append(query(l, r))
        return answer