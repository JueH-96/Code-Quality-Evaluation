from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        from math import comb
        
        N = len(nums)
        if N < 5:
            return 0
        
        # Precompute prefix frequencies
        prefix_freq = [defaultdict(int) for _ in range(N)]
        for i in range(N):
            if i > 0:
                prefix_freq[i] = prefix_freq[i-1].copy()
            prefix_freq[i][nums[i]] += 1
        
        # Precompute suffix frequencies
        suffix_freq = [defaultdict(int) for _ in range(N)]
        for i in range(N-1, -1, -1):
            if i < N-1:
                suffix_freq[i] = suffix_freq[i+1].copy()
            suffix_freq[i][nums[i]] += 1
        
        # Precompute prefix distinct sets excluding v
        prefix_distinct = [set() for _ in range(N)]
        for i in range(N):
            if i > 0:
                prefix_distinct[i] = prefix_distinct[i-1].copy()
            if nums[i] != nums[2]:
                prefix_distinct[i].add(nums[i])
        
        # Precompute suffix distinct sets excluding v
        suffix_distinct = [set() for _ in range(N)]
        for i in range(N-1, -1, -1):
            if i < N-1:
                suffix_distinct[i] = suffix_distinct[i+1].copy()
            if nums[i] != nums[2]:
                suffix_distinct[i].add(nums[i])
        
        # Precompute combinations C(n, k)
        max_n = 1000
        C = [[0] * (max_n + 1) for _ in range(5)]
        for n in range(max_n + 1):
            C[0][n] = 1
            for k in range(1, 5):
                if n >= k:
                    C[k][n] = (C[k-1][n-1] + C[k][n-1]) % MOD
                else:
                    C[k][n] = 0
        
        total = 0
        for i2 in range(2, N-3):
            v = nums[i2]
            left_v = prefix_freq[i2-1][v]
            right_v = suffix_freq[i2+1][v]
            
            left_distinct = len(prefix_distinct[i2-1])
            right_distinct = len(suffix_distinct[i2+1])
            
            union_distinct = len(prefix_distinct[i2-1].union(suffix_distinct[i2+1]))
            
            for k in range(2, 6):
                if k - 1 > left_v + right_v:
                    continue
                ways_v = 0
                for c_left in range(max(0, k-1 - right_v), min(left_v, k-1) + 1):
                    c_right = k - 1 - c_left
                    if 0 <= c_right <= right_v:
                        ways_v = (ways_v + comb(left_v, c_left) * comb(right_v, c_right)) % MOD
                remaining = 4 - (k - 1)
                if union_distinct >= remaining:
                    ways_distinct = C[remaining][union_distinct]
                else:
                    ways_distinct = 0
                total = (total + ways_v * ways_distinct) % MOD
        return total