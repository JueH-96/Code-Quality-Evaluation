import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        min_coin = min(coins)
        n = len(coins)
        
        def count(x):
            total = 0
            for mask in range(1, 1 << n):
                bits_count = bin(mask).count('1')
                current_lcm = 1
                for i in range(n):
                    if (mask >> i) & 1:
                        current_lcm = current_lcm * coins[i] // math.gcd(current_lcm, coins[i])
                        if current_lcm > x:
                            break
                if current_lcm > x:
                    continue
                if bits_count % 2 == 1:
                    total += x // current_lcm
                else:
                    total -= x // current_lcm
            return total
        
        low = 1
        high = min_coin * k
        while low < high:
            mid = (low + high) // 2
            c = count(mid)
            if c < k:
                low = mid + 1
            else:
                high = mid
        return low