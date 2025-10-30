class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
            suffix[i + 1] = suffix[i] + nums[n - i - 1]
        ans = 0
        for i in range(n):
            if prefix[i] > suffix[i + 1]:
                ans += prefix[i] - suffix[i + 1]
        return ans