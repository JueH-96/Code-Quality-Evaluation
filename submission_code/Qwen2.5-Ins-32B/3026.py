class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # If n is less than half of target, we can use the first n numbers
        if n <= target // 2:
            return (n * (n + 1) // 2) % MOD
        
        # Calculate the sum of the first half of the range
        first_half = target // 2
        sum_first_half = (first_half * (first_half + 1) // 2) % MOD
        
        # Calculate the number of elements in the second half
        second_half = n - first_half
        
        # Calculate the sum of the second half starting from target
        start_second_half = target
        end_second_half = start_second_half + second_half - 1
        sum_second_half = ((start_second_half + end_second_half) * second_half // 2) % MOD
        
        # Combine the sums of both halves
        total_sum = (sum_first_half + sum_second_half) % MOD
        
        return total_sum