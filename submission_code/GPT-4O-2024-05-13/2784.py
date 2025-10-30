class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Precompute the powers of 2 up to n-1
        power_of_2 = [1] * n
        for i in range(1, n):
            power_of_2[i] = (power_of_2[i-1] * 2) % MOD
        
        result = 0
        for i in range(n):
            max_contrib = (nums[i] * nums[i]) % MOD
            min_contrib = (nums[i] * power_of_2[i]) % MOD
            result = (result + max_contrib * min_contrib) % MOD
        
        return result