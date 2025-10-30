import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        coins.sort()
        m = coins[0]
        low, high = 1, m * k
        
        def lcm(a, b):
            return a * b // math.gcd(a, b)
        
        def count(x):
            n = len(coins)
            total_count = 0
            for mask in range(1, 1 << n):
                current_lcm = 1
                skip = False
                for j in range(n):
                    if mask & (1 << j):
                        current_lcm = lcm(current_lcm, coins[j])
                        if current_lcm > x:
                            skip = True
                            break
                if skip:
                    continue
                bits = mask.bit_count()
                sign = 1 if bits % 2 == 1 else -1
                total_count += sign * (x // current_lcm)
            return total_count
        
        while low < high:
            mid = (low + high) // 2
            if count(mid) >= k:
                high = mid
            else:
                low = mid + 1
        return low