class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        total_power = 0
        prefix_sum = 0
        
        for num in nums:
            total_power = (total_power + num * num % MOD * (num + prefix_sum) % MOD) % MOD
            prefix_sum = (prefix_sum * 2 + num) % MOD
        
        return total_power