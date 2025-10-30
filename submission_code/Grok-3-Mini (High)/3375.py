from typing import List
import math
import functools

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count_less_equal(N: int) -> int:
            n = len(coins)
            cnt = 0
            for mask in range(1, 1 << n):
                size = bin(mask).count('1')
                subset = [coins[j] for j in range(n) if (mask & (1 << j)) != 0]
                lcm_val = functools.reduce(lambda a, b: a * b // math.gcd(a, b), subset)
                count_val = N // lcm_val
                sign = (-1) ** (size - 1)
                cnt += sign * count_val
            return cnt
        
        low = 1
        high = max(coins) * k
        while low < high:
            mid = (low + high) // 2
            if count_less_equal(mid) < k:
                low = mid + 1
            else:
                high = mid
        return low