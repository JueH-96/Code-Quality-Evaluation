class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate half of target
        half = target // 2
        
        # Determine how many numbers we'll take from the first range
        from_first_range = min(n, half)
        
        # Calculate the sum of numbers from the first range
        sum_from_first_range = (from_first_range * (from_first_range + 1)) // 2
        
        # If we've filled all n slots, we're done
        if from_first_range == n:
            return sum_from_first_range % MOD
        
        # Otherwise, we need more numbers - take them from target onwards
        remaining = n - from_first_range
        
        # Calculate the sum of numbers from target to (target + remaining - 1)
        first_num = target
        last_num = target + remaining - 1
        sum_from_second_range = (remaining * (first_num + last_num)) // 2
        
        # Total sum
        total_sum = (sum_from_first_range + sum_from_second_range) % MOD
        
        return total_sum