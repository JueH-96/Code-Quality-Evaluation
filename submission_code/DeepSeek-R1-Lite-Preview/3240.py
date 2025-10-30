class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def sum_set_bits(num, x):
            total = 0
            p = 1
            while (1 << p) <= (1 << 60):  # up to 60 bits
                if p % x == 0:
                    cycle = 1 << p
                    half_cycle = 1 << (p - 1)
                    full_cycles = num // cycle
                    set_bits = full_cycles * half_cycle
                    remaining = num % cycle
                    set_bits += max(0, remaining - half_cycle + 1)
                    total += set_bits
                p += 1
            return total
        
        low = 1
        high = 10**16  # Upper bound based on k and x constraints
        answer = 0
        while low <= high:
            mid = (low + high) // 2
            s = sum_set_bits(mid, x)
            if s <= k:
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        return answer