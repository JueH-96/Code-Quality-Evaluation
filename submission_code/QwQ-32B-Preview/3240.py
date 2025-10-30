class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def sum_prices(num, x):
            total = 0
            for i in range(60):
                if (i + 1) % x == 0:
                    bit = 1 << i
                    higher = num // (bit * 2)
                    lower = num % (bit * 2)
                    if num & bit:
                        total += higher * bit + max(lower - bit + 1, 0)
                    else:
                        total += higher * bit
            return total
        
        low = 1
        high = 10**18
        while low <= high:
            mid = (low + high) // 2
            if sum_prices(mid, x) <= k:
                low = mid + 1
            else:
                high = mid - 1
        return high