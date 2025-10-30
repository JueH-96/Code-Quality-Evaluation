from typing import List
import itertools
from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        # Pre-process the coins to remove redundant ones
        coins = sorted(coins)
        filtered = []
        for c in coins:
            keep = True
            for f in filtered:
                if c % f == 0:
                    keep = False
                    break
            if keep:
                filtered.append(c)
        
        if not filtered:
            return 0  # This case is impossible per problem constraints
        
        smallest = filtered[0]
        low = 1
        high = smallest * k
        
        def lcm(a, b):
            return a * b // gcd(a, b) if a and b else 0
        
        def count(x):
            res = 0
            n = len(filtered)
            for i in range(1, n + 1):
                for subset in itertools.combinations(filtered, i):
                    current_lcm = 1
                    for num in subset:
                        current_lcm = lcm(current_lcm, num)
                        if current_lcm > x:
                            break
                    if current_lcm > x:
                        continue
                    cnt = x // current_lcm
                    if i % 2 == 1:
                        res += cnt
                    else:
                        res -= cnt
            return res
        
        while low < high:
            mid = (low + high) // 2
            c = count(mid)
            if c >= k:
                high = mid
            else:
                low = mid + 1
        
        return low