class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10 ** 9 + 7
        
        # Sort the array in ascending order
        nums.sort()
        
        total_power = 0
        
        for i in range(n):
            max_val = nums[i]
            min_val = nums[i]
            
            # Calculate the power of the current group
            power = (max_val ** 2) * min_val
            total_power = (total_power + power) % MOD
            
            # Calculate the power of all subgroups that include the current element
            for j in range(i + 1, n):
                max_val = max(max_val, nums[j])
                min_val = min(min_val, nums[j])
                power = (max_val ** 2) * min_val
                total_power = (total_power + power) % MOD
        
        return total_power