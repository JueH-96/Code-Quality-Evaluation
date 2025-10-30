class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Numbers we can use from 1 to k//2
        first_part = min(n, k//2)
        # Sum of first_part numbers: 1 + 2 + ... + first_part
        sum_first = first_part * (first_part + 1) // 2
        
        # If we need more numbers
        if n > first_part:
            remaining = n - first_part
            # We use k, k+1, ..., k+remaining-1
            sum_remaining = remaining * k + remaining * (remaining - 1) // 2
            return sum_first + sum_remaining
        else:
            return sum_first