from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        
        # Precompute the score[i][j]
        score = [[0] * n for _ in range(n)]
        for i in range(n):
            score[i][i] = nums[i]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                score[i][j] = score[i][j-1] ^ score[i+1][j]
        
        # Precompute max_from_i[i][j]: maximum score for subarrays starting at i and ending at j
        max_from_i = [[0] * n for _ in range(n)]
        for i in range(n):
            current_max = float('-inf')
            for j in range(i, n):
                current_max = max(current_max, score[i][j])
                max_from_i[i][j] = current_max
        
        # Precompute suffix_max[R][L]: maximum of max_from_i[i][R] for i in [L..R]
        suffix_max = [[0] * n for _ in range(n)]
        for R in range(n):
            current_max = float('-inf')
            for L in range(R, -1, -1):
                current_max = max(current_max, max_from_i[L][R])
                suffix_max[R][L] = current_max
        
        # Process each query
        res = []
        for l, r in queries:
            res.append(suffix_max[r][l])
        
        return res