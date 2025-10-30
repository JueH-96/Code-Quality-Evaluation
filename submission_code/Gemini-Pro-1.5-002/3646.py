class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        dp = [0] * (n + 1)
        sums = [0] * 100001
        ans = 0
        for num in nums:
            new_sum = sums[num]
            if num > 0:
                new_sum = (new_sum + sums[num - 1]) % mod
            if num < 100000:
                new_sum = (new_sum + sums[num + 1]) % mod
            new_sum = (new_sum + 1) % mod
            sums[num] = new_sum
            ans = (ans + new_sum) % mod

        return ans