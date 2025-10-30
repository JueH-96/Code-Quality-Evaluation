class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def total_price(num):
            total = 0
            j = 0
            while (1 << j) <= num:
                if (j + 1) % x == 0:
                    period = 1 << (j + 1)
                    full_cycles = (num + 1) // period
                    count = full_cycles * (1 << j)
                    remainder = (num + 1) % period
                    if remainder > (1 << j):
                        count += remainder - (1 << j)
                    total += count
                j += 1
            return total
        
        low, high = 0, 10**18
        while low < high:
            mid = (low + high) // 2
            if total_price(mid) <= k:
                low = mid + 1
            else:
                high = mid
        return low - 1