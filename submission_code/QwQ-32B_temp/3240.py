class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_total(N):
            total = 0
            if N == 0:
                return 0
            max_bit = N.bit_length()
            for i in range(1, max_bit + 1):
                if i % x != 0:
                    continue
                bit_val = 1 << (i - 1)
                cycle = bit_val << 1
                full_cycles = (N + 1) // cycle
                remainder = (N + 1) % cycle
                cnt = full_cycles * bit_val + max(0, remainder - bit_val)
                total += cnt
            return total
        
        left = 0
        right = 1
        while compute_total(right) <= k:
            right *= 2
        
        while left < right:
            mid = (left + right + 1) // 2
            s = compute_total(mid)
            if s <= k:
                left = mid
            else:
                right = mid - 1
        
        return left