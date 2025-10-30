class Solution:
    def kthCharacter(self, k: int) -> str:
        if k == 0:
            return ''
        
        # Determine the minimal m such that 2^m >= k
        m = 0
        current = 1
        while current < k:
            current *= 2
            m += 1
        
        shift_count = 0
        current_m = m
        current_k = k
        
        while current_m > 0:
            mid = 2 ** (current_m - 1)
            if current_k > mid:
                shift_count += 1
                current_k -= mid
            current_m -= 1
        
        return chr(ord('a') + shift_count)