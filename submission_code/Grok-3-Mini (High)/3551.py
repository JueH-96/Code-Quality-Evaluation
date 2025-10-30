import math
from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def get_S(d):
            if d == 0:
                return [0]
            pos = [k for k in range(11) if (d & (1 << k)) != 0]  # Bit positions set in d
            num_bits = len(pos)
            num_subsets = 1 << num_bits
            S = []
            for mask in range(num_subsets):
                subset_val = 0
                for b in range(num_bits):
                    if (mask & (1 << b)) != 0:
                        bit_pos = pos[b]
                        subset_val |= (1 << bit_pos)
                S.append(subset_val)
            return S
        
        n = len(nums)
        # Precompute sparse tables for each length
        st_per_len = [None] * (n + 1)  # Index by length, from 1 to n
        
        for len_val in range(1, n + 1):
            d = len_val - 1
            S_d = get_S(d)
            num_subarrays = n - len_val + 1
            scores = [0] * num_subarrays
            for start in range(num_subarrays):
                score_val = 0
                for i in S_d:
                    score_val ^= nums[start + i]
                scores[start] = score_val
            # Build sparse table for scores
            m = len(scores)
            logm = math.floor(math.log2(m)) + 1 if m > 0 else 0
            st = [[0 for _ in range(logm)] for _ in range(m)]
            for i in range(m):
                st[i][0] = scores[i]
            for k in range(1, logm):
                len_k = 1 << k
                for i in range(m - len_k + 1):
                    st[i][k] = max(st[i][k - 1], st[i + (1 << (k - 1))][k - 1])
            st_per_len[len_val] = st
        
        # Process each query
        answer = []
        for query in queries:
            ql, qr = query[0], query[1]
            max_score = 0  # XOR scores are non-negative
            for len_val in range(1, n + 1):
                low_start = ql
                high_start = min(qr - len_val + 1, n - len_val)
                if low_start <= high_start:
                    st = st_per_len[len_val]
                    L = low_start
                    R = high_start
                    len_range = R - L + 1
                    K = math.floor(math.log2(len_range))
                    max_len_val = max(st[L][K], st[R - (1 << K) + 1][K])
                    if max_len_val > max_score:
                        max_score = max(max_score, max_len_val)
            answer.append(max_score)
        
        return answer