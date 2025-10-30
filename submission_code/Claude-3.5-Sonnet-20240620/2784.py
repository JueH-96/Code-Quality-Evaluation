class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        result = 0
        prefix_sum = 0
        
        for i, num in enumerate(nums):
            result = (result + (num * num % MOD) * num) % MOD
            result = (result + (num * num % MOD) * prefix_sum) % MOD
            prefix_sum = (prefix_sum * 2 + num) % MOD
        
        return result