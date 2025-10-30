class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 0
        high = 1
        
        while self.total_price(high, x) <= k:
            low = high
            high *= 2
        
        ans = low
        while low <= high:
            mid = (low + high) // 2
            p_mid = self.total_price(mid, x)
            if p_mid <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans

    def total_price(self, n: int, x: int) -> int:
        if n == 0:
            return 0
        total = 0
        j = x
        while (1 << (j - 1)) <= n:
            period = 1 << j
            full_cycles = n // period
            remainder = n % period
            count_j = full_cycles * (1 << (j - 1))
            if remainder >= (1 << (j - 1)):
                count_j += remainder - (1 << (j - 1)) + 1
            total += count_j
            j += x
        return total