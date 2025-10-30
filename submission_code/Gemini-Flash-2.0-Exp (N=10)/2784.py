class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        MOD = 10**9 + 7
        ans = 0
        pow2 = 1
        for i in range(n):
            ans = (ans + (pow2 * nums[i] * nums[i] % MOD * nums[i] % MOD) % MOD) % MOD
            if i < n - 1:
                ans = (ans + (pow2 * nums[i] * nums[i] % MOD * sum(nums[i+1:]) % MOD) % MOD) % MOD
            pow2 = (pow2 * 2) % MOD
        return ans