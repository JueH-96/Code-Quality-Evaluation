class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # Find the number of elements that can be placed before reaching target/2
        k = min(target // 2, n)
        
        # Sum of first k elements
        sum_k = (k * (k + 1)) // 2
        
        if n <= k:
            return sum_k % MOD
        
        # Remaining elements start from target
        remaining = n - k
        sum_remaining = (target + target + remaining - 1) * remaining // 2
        
        total_sum = sum_k + sum_remaining
        return total_sum % MOD