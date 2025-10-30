class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        result = 0
        
        for i in range(n):
            for j in range(i, n):
                min_val = nums[i]
                max_val = nums[j]
                power = (max_val * max_val % MOD * min_val) % MOD
                
                if i == j:
                    result = (result + power) % MOD
                else:
                    count = pow(2, j - i - 1, MOD)
                    result = (result + power * count) % MOD
        
        return result