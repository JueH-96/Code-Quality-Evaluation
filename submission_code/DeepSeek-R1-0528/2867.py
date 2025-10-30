class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        ones = [i for i, x in enumerate(nums) if x == 1]
        if not ones:
            return 0
        res = 1
        for i in range(len(ones) - 1):
            res = (res * (ones[i+1] - ones[i])) % mod
        return res