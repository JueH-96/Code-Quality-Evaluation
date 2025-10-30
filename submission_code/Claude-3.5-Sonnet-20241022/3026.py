class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # For numbers less than target/2, we can include them
        # For numbers >= target/2, we need to skip numbers that sum to target
        mid = target // 2
        
        if n <= mid:
            # If n is small enough, just take first n numbers
            return (n * (n + 1) // 2) % MOD
            
        # Count how many numbers we need after mid
        remaining = n - mid
        
        # First mid numbers sum: 1 + 2 + ... + mid
        first_sum = (mid * (mid + 1) // 2)
        
        # For remaining numbers, start from target
        # to avoid any sum equal to target
        start = target
        remaining_sum = (remaining * (2 * start + (remaining - 1))) // 2
        
        return (first_sum + remaining_sum) % MOD