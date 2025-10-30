from typing import List
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        subsets = []
        n = len(coins)
        for mask in range(1, 1 << n):
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(coins[i])
            # Compute LCM of the subset
            lcm = 1
            for num in subset:
                lcm = lcm * num // gcd(lcm, num)
            size = len(subset)
            sign = 1 if size % 2 == 1 else -1
            subsets.append((sign, lcm))
        
        min_coin = min(coins)
        low = 1
        high = min_coin * k
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            count = 0
            for sign, lcm in subsets:
                if lcm > mid:
                    continue
                m = mid // lcm
                count += sign * m
            if count >= k:
                high = mid - 1
                answer = mid
            else:
                low = mid + 1
        return answer