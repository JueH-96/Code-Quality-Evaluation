class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        nums = [0] + sorted(x for x in nums if x > 0) + [0]
        sums = [0]*(len(nums))
        for i in range(1, len(nums)):
            sums[i] = sums[i-2] + nums[i]
        res = sum(sums[i] for i in range(len(nums)) if nums[i] > 0) % MOD
        for pos, x in queries:
            if nums[pos+1] > 0:
                res = (res - sums[pos+1] + sums[pos-1] + max(0, x)) % MOD
            else:
                res = (res + max(0, x)) % MOD
            nums[pos+1] = x
        return res