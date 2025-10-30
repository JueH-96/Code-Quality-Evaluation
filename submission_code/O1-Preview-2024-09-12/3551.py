class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        max_score_in_interval = [[0]*n for _ in range(n)]
        # Precompute Pascal's triangle mod 2
        pascal_mod2 = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n+1):
            pascal_mod2[i][0] = 1
            for j in range(1, i+1):
                pascal_mod2[i][j] = (pascal_mod2[i-1][j-1] + pascal_mod2[i-1][j]) %2
        
        # Precompute positions where C[n-1][i] mod 2 == 1
        len_to_pos = [[] for _ in range(n+1)]
        for length in range(1, n+1):
            for pos in range(length):
                if pascal_mod2[length-1][pos]:
                    len_to_pos[length].append(pos)
        
        # Compute prefix XOR for fast computation
        prefix_xor = [0]*(n+1)
        for i in range(n):
            prefix_xor[i+1] = prefix_xor[i] ^ nums[i]
        
        # Precompute maximum scores for all intervals
        for L in range(n):
            max_score = float('-inf')
            score_dict = {}
            for length in range(1, n - L +1):
                R = L + length -1
                # Compute score for nums[L..R]
                score = 0
                for pos in len_to_pos[length]:
                    idx = L + pos
                    score ^= nums[idx]
                max_score = max(max_score, score)
                max_score_in_interval[L][R] = max_score
        
        # Answer queries
        answer = []
        for l_i, r_i in queries:
            answer.append(max_score_in_interval[l_i][r_i])
        return answer