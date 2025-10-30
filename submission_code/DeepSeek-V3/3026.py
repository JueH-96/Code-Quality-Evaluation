class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Determine the maximum number we can use without causing a sum equal to target
        # We can use numbers from 1 to (target // 2 - 1) and then skip the numbers that would pair with them to sum to target
        # After that, we can use numbers starting from target
        # So, the first part is the sum of the first k numbers, where k is the number of numbers less than target // 2
        k = min(n, (target - 1) // 2)
        sum_first_k = k * (k + 1) // 2
        remaining = n - k
        # The remaining numbers start from target
        sum_remaining = remaining * (2 * target + remaining - 1) // 2
        total = (sum_first_k + sum_remaining) % MOD
        return total