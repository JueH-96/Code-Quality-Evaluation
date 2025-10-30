class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def calculate_sum(num):
            total = 0
            i = x
            while True:
                if i > 60:
                    break
                pow_i = 1 << (i - 1)
                cycle = pow_i * 2
                full_cycles = (num + 1) // cycle
                count = full_cycles * pow_i
                remainder = (num + 1) % cycle
                if remainder > pow_i:
                    count += remainder - pow_i
                total += count
                i += x
            return total
        
        low = 0
        high = 1 << 60  # A very large number to cover all possible cases
        best = 0
        while low <= high:
            mid = (low + high) // 2
            s = calculate_sum(mid)
            if s <= k:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best