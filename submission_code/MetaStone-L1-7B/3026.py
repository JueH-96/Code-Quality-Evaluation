class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        m = (target - 1) // 2
        sum_m = m * (m + 1) // 2
        if target % 2 == 0:
            sum_m += target // 2
        rem = n - (m + 1)
        if rem > 0:
            sum_m += rem * (2 * target + rem - 1) // 2
        return sum_m % MOD