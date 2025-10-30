from typing import List
import math

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def count_unique(x: int) -> int:
            total = 0
            n = len(coins)
            for mask in range(1, 1 << n):
                bits = bin(mask).count('1')
                subset = [coins[i] for i in range(n) if mask & (1 << i)]
                lcm_subset = math.lcm(*subset)
                if bits % 2 == 1:
                    total += x // lcm_subset
                else:
                    total -= x // lcm_subset
            return total
        
        coins = sorted(coins)
        low = coins[0]
        high = low * k  # Upper bound can be adjusted based on min coin and k
        while low < high:
            mid = (low + high) // 2
            cnt = count_unique(mid)
            if cnt < k:
                low = mid + 1
            else:
                high = mid
        return low