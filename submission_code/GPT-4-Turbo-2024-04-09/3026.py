class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        if target % 2 == 0:
            # If target is even, we cannot have target//2 in the array
            half_target = target // 2
            if n <= half_target - 1:
                # If n is less than half_target, we can use the first n numbers
                return n * (n + 1) // 2 % MOD
            else:
                # We need to skip the number half_target
                sum_below_half_target = (half_target - 1) * half_target // 2
                sum_above_half_target = (half_target + 1 + half_target + n - 1) * n // 2
                return (sum_below_half_target + sum_above_half_target) % MOD
        else:
            # If target is odd, there's no restriction based on half of target
            return n * (n + 1) // 2 % MOD