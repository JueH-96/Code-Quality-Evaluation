class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        mod = 10 ** 9 + 7
        total_or = nums[0]
        n = len(nums)
        for i in range(k, n):
            total_or |= nums[i]
        ans = pow(total_or, 2, mod)
        for i in range(1, k):
            ans = (ans + nums[i] * nums[i]) % mod
        return ans