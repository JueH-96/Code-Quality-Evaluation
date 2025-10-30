class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 0
        high = 10**18
        best = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = self.compute_sum(mid, x)
            if current_sum <= k:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best
    
    def compute_sum(self, num: int, x: int) -> int:
        total = 0
        i = x
        while i <= 60:
            cycle = 1 << i
            full_cycles = num // cycle
            count = full_cycles * (1 << (i - 1))
            remainder = num % cycle
            if remainder >= (1 << (i - 1)):
                count += remainder - (1 << (i - 1)) + 1
            total += count
            i += x
        return total