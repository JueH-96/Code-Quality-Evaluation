import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        subsets = []
        for mask in range(1, 1 << n):
            elements = [coins[i] for i in range(n) if (mask & (1 << i))]
            size = len(elements)
            lcm = 1
            for num in elements:
                gcd_val = math.gcd(lcm, num)
                lcm = (lcm // gcd_val) * num
            sign = (-1) ** (size + 1)
            subsets.append((lcm, sign))
        
        left = 1
        right = 1 << 60  # A sufficiently large upper bound
        answer = 0
        while left <= right:
            mid = (left + right) // 2
            cnt = 0
            for (l, s) in subsets:
                cnt += s * (mid // l)
            if cnt >= k:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer