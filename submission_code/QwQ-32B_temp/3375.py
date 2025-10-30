import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        subsets = []
        
        # Precompute all non-empty subsets' LCM and their sizes
        for mask in range(1, 1 << n):
            coins_in_subset = []
            for i in range(n):
                if mask & (1 << i):
                    coins_in_subset.append(coins[i])
            current_lcm = 1
            for c in coins_in_subset:
                current_lcm = self.lcm(current_lcm, c)
            m = len(coins_in_subset)
            subsets.append((current_lcm, m))
        
        def compute_count(x):
            total = 0
            for lcm_val, m in subsets:
                cnt = x // lcm_val
                if m % 2 == 1:
                    total += cnt
                else:
                    total -= cnt
            return total
        
        low = 1
        high = max(coins) * k
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            cnt = compute_count(mid)
            if cnt >= k:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
    
    def lcm(self, a: int, b: int) -> int:
        return a * b // math.gcd(a, b)