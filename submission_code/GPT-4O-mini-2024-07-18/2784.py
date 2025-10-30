class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        total_power = 0
        current_sum = 0
        
        for i in range(n):
            max_strength = nums[i]
            # Calculate the contribution of nums[i] as the maximum strength
            # The number of subsets that can be formed with nums[0] to nums[i] is 2^i
            # The contribution of nums[i] as max is max_strength^2 * min_strength
            # where min_strength can be any of the previous elements or nums[i] itself
            
            # The contribution of this max_strength
            contribution = (max_strength * max_strength) % MOD
            
            # The number of ways to choose elements before nums[i]
            # is 2^i (all subsets of the first i elements)
            current_sum = (current_sum * 2 + 1) % MOD
            
            # Add the contribution of this max_strength to the total power
            total_power = (total_power + contribution * current_sum) % MOD
        
        return total_power