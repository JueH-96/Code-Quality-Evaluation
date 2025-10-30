class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(num):
            total = 0
            p = x
            while (1 << (p-1)) <= num:
                cycle = 1 << p
                full_cycles = num // cycle
                total += full_cycles * (1 << (p-1))
                remainder = num % cycle
                total += max(0, remainder - (1 << (p-1)) + 1)
                p += x
            return total
        
        left, right = 1, 2
        while count_set_bits(right) <= k:
            right *= 2
        
        result = 0
        while left <= right:
            mid = (left + right) // 2
            current = count_set_bits(mid)
            if current <= k:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result