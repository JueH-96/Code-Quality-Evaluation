class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        
        result = 0
        for i in range(n):
            for j in range(i + 1):
                if j == i:
                    result = (result + nums[j] ** 3) % MOD
                else:
                    result = (result + (nums[j] * (nums[i] ** 2) % MOD) * pow2[i - j - 1]) % MOD
        return result