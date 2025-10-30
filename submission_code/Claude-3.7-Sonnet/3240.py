class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_prices(n):
            """Calculate the sum of prices for numbers from 1 to n."""
            total = 0
            j = 1
            while (1 << (j - 1)) <= n:
                if j % x == 0:
                    cycle = 1 << j
                    half_cycle = cycle >> 1
                    complete_cycles = n // cycle
                    
                    total += complete_cycles * half_cycle
                    
                    remaining = n % cycle
                    if remaining >= half_cycle:
                        total += remaining - half_cycle + 1
                j += 1
            return total
        
        left, right = 1, 10**18
        while left <= right:
            mid = (left + right) >> 1
            if count_prices(mid) <= k:
                left = mid + 1
            else:
                right = mid - 1
        return right