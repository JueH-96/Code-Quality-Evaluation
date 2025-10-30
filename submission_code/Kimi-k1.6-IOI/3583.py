import math
from typing import List
from bisect import bisect_right
from collections import Counter

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        
        freq = Counter(nums)
        unique_nums = list(freq.keys())
        max_d = max(nums)
        cnt = [0] * (max_d + 1)
        
        for x in unique_nums:
            divisors = set()
            for i in range(1, math.isqrt(x) + 1):
                if x % i == 0:
                    divisors.add(i)
                    divisors.add(x // i)
            for d in divisors:
                if d <= max_d:
                    cnt[d] += freq[x]
        
        exact = [0] * (max_d + 1)
        for d in range(max_d, 0, -1):
            total = cnt[d] * (cnt[d] - 1) // 2
            sum_multiples = 0
            m = 2 * d
            while m <= max_d:
                sum_multiples += exact[m]
                m += d
            exact[d] = total - sum_multiples
        
        gcd_list = []
        for d in range(1, max_d + 1):
            if exact[d] > 0:
                gcd_list.append((d, exact[d]))
        
        gcd_list.sort()
        prefix = []
        current_sum = 0
        for d, cnt in gcd_list:
            current_sum += cnt
            prefix.append(current_sum)
        
        res = []
        for q in queries:
            idx = bisect_right(prefix, q)
            res.append(gcd_list[idx][0])
        
        return res