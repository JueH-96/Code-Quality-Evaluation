class Solution:
    from math import gcd
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        from math import gcd
        from functools import reduce

        n = len(coins)
        lcm_list = []
        sign_list = []

        def lcm(a,b):
            return a*b//gcd(a,b)
        from itertools import combinations

        for size in range(1, n+1):
            subsets = combinations(range(n), size)
            for subset in subsets:
                lcm_value = coins[subset[0]]
                for idx in subset[1:]:
                    lcm_value = lcm(lcm_value, coins[idx])
                lcm_list.append(lcm_value)
                sign_list.append((-1) ** (size +1))

        def count(x):
            total = 0
            for lcm_value, sign in zip(lcm_list, sign_list):
                total += sign * (x // lcm_value)
            return total

        left = 1
        right = 1 << 60
        while left < right:
            mid = (left + right) // 2
            if count(mid) >= k:
                right = mid
            else:
                left = mid +1
        return left