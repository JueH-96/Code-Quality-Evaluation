from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Precompute xor_score[a][k] where k is (b-a) for subarray [a, b]
        xor_score = [[0] * (n - a) for a in range(n)]
        
        for a in range(n):
            for k in range(n - a):
                current_xor = 0
                # Iterate through all i where (i & k) == i
                i = 0
                while True:
                    if (i & k) == i:
                        if a + i >= n:
                            break
                        current_xor ^= nums[a + i]
                    if i >= k:
                        break
                    i += 1
                xor_score[a][k] = current_xor
        
        # Precompute max_xor[a][k]: maximum xor_score[a][0..k]
        max_xor = [[0] * (n - a) for a in range(n)]
        for a in range(n):
            max_val = 0
            for k in range(n - a):
                if k == 0:
                    max_val = xor_score[a][k]
                else:
                    max_val = max(max_val, xor_score[a][k])
                max_xor[a][k] = max_val
        
        # Process each query
        result = []
        for l, r in queries:
            max_query = 0
            for a in range(l, r + 1):
                max_k = r - a
                if max_k >= 0:
                    current_max = max_xor[a][max_k]
                    if current_max > max_query:
                        max_query = current_max
            result.append(max_query)
        
        return result