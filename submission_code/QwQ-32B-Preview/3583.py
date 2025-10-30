from typing import List
from math import gcd
from collections import Counter
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_num = max(nums)
        num_count = Counter(nums)
        
        # Initialize f(d) for all d from 1 to max_num
        f = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            count_c_d = sum(num_count[x] for x in num_count if x % d == 0)
            f[d] = count_c_d * (count_c_d - 1) // 2
        
        # Adjust f(d) to count only pairs with GCD exactly d
        for d in range(max_num, 0, -1):
            multiple = d + d
            while multiple <= max_num:
                f[d] -= f[multiple]
                multiple += d
        
        # Build prefix sum array
        prefix = [0] * (max_num + 1)
        cumulative = 0
        for d in range(1, max_num + 1):
            cumulative += f[d]
            prefix[d] = cumulative
        
        # Answer each query
        answer = []
        for q in queries:
            # Find the smallest d such that prefix[d] >= q + 1
            left = 1
            right = max_num
            res = -1
            while left <= right:
                mid = (left + right) // 2
                if prefix[mid] >= q + 1:
                    res = mid
                    right = mid - 1
                else:
                    left = mid + 1
            answer.append(res)
        
        return answer