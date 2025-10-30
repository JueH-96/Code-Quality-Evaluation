class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        k = target // 2
        m = k + 1  # Maximum numbers that can be included without violating the condition
        if n <= m:
            if n <= k:
                # Sum of first n natural numbers
                total_sum = n * (n + 1) // 2
            else:
                # Sum of first k natural numbers plus target
                sum_k = k * (k + 1) // 2
                total_sum = sum_k + target
        else:
            # Sum of first k natural numbers plus target plus the next (n - m) numbers starting from target + 1
            sum_k = k * (k + 1) // 2
            p = n - m
            # Sum from target + 1 to target + p
            a = target + 1
            b = target + p
            sum_p = p * (a + b) // 2
            total_sum = sum_k + target + sum_p
        return total_sum % MOD