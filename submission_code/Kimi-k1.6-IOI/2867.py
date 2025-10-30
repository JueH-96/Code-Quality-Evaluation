class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones = [i for i, num in enumerate(nums) if num == 1]
        n = len(ones)
        if n == 0:
            return 0
        if n == 1:
            return 1
        result = 1
        for i in range(n - 1):
            result = result * (ones[i+1] - ones[i]) % MOD
        return result