import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        max_coin = max(coins)
        low, high = 1, k * max_coin
        
        def count_amounts(x: int) -> int:
            n = len(coins)
            total = 0
            for mask in range(1, 1 << n):
                lcm_val = 1
                cnt = 0
                skip_subset = False
                for j in range(n):
                    if mask & (1 << j):
                        cnt += 1
                        g = math.gcd(lcm_val, coins[j])
                        next_lcm = lcm_val * coins[j] // g
                        if next_lcm > x:
                            skip_subset = True
                            break
                        lcm_val = next_lcm
                if skip_subset:
                    continue
                if cnt % 2 == 1:
                    total += x // lcm_val
                else:
                    total -= x // lcm_val
            return total
        
        while low < high:
            mid = (low + high) // 2
            if count_amounts(mid) >= k:
                high = mid
            else:
                low = mid + 1
                
        return low