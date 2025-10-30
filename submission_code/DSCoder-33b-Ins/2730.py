class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
            suffix[n - i - 1] = suffix[n - i] | nums[n - i - 1]
        res = 0
        for i in range(n):
            res = max(res, prefix[i] | nums[i] << k | suffix[i + 1])
        return res