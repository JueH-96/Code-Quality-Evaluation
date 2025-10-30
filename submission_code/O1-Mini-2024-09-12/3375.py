class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        from math import gcd
        from functools import reduce

        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def count(mid):
            total = 0
            n = len(coins)
            for mask in range(1, 1 << n):
                bits = bin(mask).count('1')
                subset = []
                current_lcm = 1
                for i in range(n):
                    if mask & (1 << i):
                        current_lcm = lcm(current_lcm, coins[i])
                        if current_lcm > mid:
                            break
                else:
                    if bits % 2 == 1:
                        total += mid // current_lcm
                    else:
                        total -= mid // current_lcm
            return total

        low = 1
        high = min(coins) * k
        while low < high:
            mid = (low + high) // 2
            c = count(mid)
            if c < k:
                low = mid + 1
            else:
                high = mid
        return low