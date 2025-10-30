class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        m = (k - 1) // 2
        take_from_start = min(n, m)
        total = take_from_start * (take_from_start + 1) // 2
        remaining = n - take_from_start
        
        if remaining > 0 and k % 2 == 0:
            total += k // 2
            remaining -= 1
        
        if remaining > 0:
            total += remaining * k + remaining * (remaining - 1) // 2
        
        return total