from typing import List
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free(n: int) -> int:
            res = 1
            i = 2
            while i * i <= n:
                if n % i == 0:
                    cnt = 0
                    while n % i == 0:
                        cnt += 1
                        n //= i
                    if cnt % 2 == 1:
                        res *= i
                i += 1
            if n > 1:
                res *= n
            return res
        
        groups = defaultdict(int)
        max_single = 0
        for num in nums:
            sf = square_free(num)
            groups[sf] += num
            if num > max_single:
                max_single = num
        
        max_group = max(groups.values(), default=0)
        return max(max_group, max_single)