class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = [i for i in range(1, n+1)]
        return sum(nums) % MOD