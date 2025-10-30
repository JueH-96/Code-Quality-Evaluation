class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # We can safely include numbers from 1 to floor(k/2)
        safe_count = min(n, k // 2)
        
        # Sum of these safe small numbers
        sum_safe = (safe_count * (safe_count + 1)) // 2
        
        # If we need more numbers, include numbers k and above
        if safe_count < n:
            remaining_count = n - safe_count
            # Sum of arithmetic sequence: k, k+1, k+2, ..., k+(remaining_count-1)
            sum_remaining = remaining_count * k + (remaining_count * (remaining_count - 1)) // 2
            return sum_safe + sum_remaining
        
        return sum_safe