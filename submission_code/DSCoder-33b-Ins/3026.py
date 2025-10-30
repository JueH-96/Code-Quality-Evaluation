class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = list(range(1, target + 1))
        if len(nums) >= n:
            return sum(nums[:n]) % MOD
        else:
            nums += list(range(target + 1, target + n - len(nums) + 1))
            return sum(nums) % MOD