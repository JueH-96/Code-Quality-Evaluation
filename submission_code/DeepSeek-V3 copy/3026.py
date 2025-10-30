class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Determine the maximum number we can take without causing a pair sum to be target
        # We can take numbers from 1 to k, where k is the largest number such that for any i < j <= k, i + j != target
        # The largest k is min(target // 2, n)
        k = min((target - 1) // 2, n)
        # Sum of the first k numbers is k*(k+1)//2
        sum_k = k * (k + 1) // 2
        # Remaining numbers to take are n - k
        remaining = n - k
        # Start from target and take the next 'remaining' numbers
        # The sum is target + (target+1) + ... + (target + remaining - 1)
        # Which is remaining * target + remaining * (remaining - 1) // 2
        sum_remaining = remaining * target + remaining * (remaining - 1) // 2
        total = (sum_k + sum_remaining) % MOD
        return total