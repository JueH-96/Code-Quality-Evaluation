import math
from bisect import bisect_right
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        max_d = 5 * 10**4  # Maximum possible value in nums is 5*10^4
        cnt = [0] * (max_d + 1)
        
        # Compute the count of each divisor for elements in nums
        for x in nums:
            divisors = set()
            sqrt_x = int(math.isqrt(x))
            for i in range(1, sqrt_x + 1):
                if x % i == 0:
                    divisors.add(i)
                    divisors.add(x // i)
            for d in divisors:
                cnt[d] += 1
        
        # Calculate f[d] (number of pairs divisible by d)
        f = [0] * (max_d + 1)
        for d in range(1, max_d + 1):
            m = cnt[d]
            f[d] = m * (m - 1) // 2
        
        # Calculate count[d] using inclusion-exclusion from high to low
        count = [0] * (max_d + 1)
        for d in range(max_d, 0, -1):
            sum_multiples = 0
            m = 2 * d
            while m <= max_d:
                sum_multiples += count[m]
                m += d
            count[d] = f[d] - sum_multiples
        
        # Collect and sort divisors with non-zero counts
        sorted_ds = [d for d in range(1, max_d + 1) if count[d] > 0]
        sorted_ds.sort()
        
        # Compute prefix sums
        prefix = []
        current_sum = 0
        for d in sorted_ds:
            current_sum += count[d]
            prefix.append(current_sum)
        
        # Answer each query using binary search on prefix sums
        answer = []
        for q in queries:
            idx = bisect_right(prefix, q)
            answer.append(sorted_ds[idx])
        
        return answer