from typing import List
from math import gcd
from collections import defaultdict
from functools import lru_cache

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        @lru_cache(None)
        def gcd_of_list(lst):
            g = lst[0]
            for num in lst[1:]:
                g = gcd(g, num)
            return g
        
        n = len(nums)
        subseq_gcd_count = defaultdict(int)
        
        for i in range(1, 1 << n):
            subseq = [nums[j] for j in range(n) if i & (1 << j)]
            g = gcd_of_list(tuple(subseq))
            subseq_gcd_count[g] += 1
        
        result = 0
        for g, count in subseq_gcd_count.items():
            result = (result + count * count) % MOD
        
        return result