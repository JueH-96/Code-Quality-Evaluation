class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # We can use numbers from 1 to k//2
        # Then we need to use numbers from k onwards
        
        if n <= k // 2:
            # If n is small enough, we can just use 1, 2, ..., n
            return n * (n + 1) // 2
        else:
            # Use numbers 1 to k//2, then k, k+1, ...
            first_part = k // 2
            sum_first = first_part * (first_part + 1) // 2
            
            # How many more numbers we need
            remaining = n - first_part
            
            # These will be k, k+1, k+2, ..., k+remaining-1
            sum_remaining = remaining * k + remaining * (remaining - 1) // 2
            
            return sum_first + sum_remaining