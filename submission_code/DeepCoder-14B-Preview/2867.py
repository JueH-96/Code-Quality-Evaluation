MOD = 10**9 + 7

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        ones = [i for i, x in enumerate(nums) if x == 1]
        if not ones:
            return 0
        if len(ones) == 1:
            return 1
        product = 1
        for i in range(len(ones) - 1):
            product = (product * (ones[i+1] - ones[i])) % MOD
        return product