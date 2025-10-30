class Solution:
    def kthCharacter(self, k: int) -> str:
        if k == 1:
            return 'a'
        
        L = 1
        n = 0
        while L < k:
            L *= 2
            n += 1
        
        shift_count = 0
        for _ in range(n):
            half = L // 2
            if k <= half:
                L = half
            else:
                k -= half
                shift_count += 1
                L = half
        
        shift_count %= 26
        return chr(ord('a') + shift_count)