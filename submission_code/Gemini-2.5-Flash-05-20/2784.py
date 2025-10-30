from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()  # Sort the array in non-decreasing order
        n = len(nums)
        
        total_power_sum = 0
        
        # prefix_min_sum stores the sum of minimums of all non-empty subsets of nums[0...i-1].
        # In the loop for nums[i], this variable holds P_i (as defined in the thought process).
        # P_0 is 0.
        prefix_min_sum = 0 
        
        for i in range(n):
            current_num = nums[i]
            
            # The contribution of current_num when it's the maximum element in a group:
            # It's (current_num^2) * (sum of minimums of groups where current_num is max).
            # The sum of minimums is:
            # 1. current_num itself (for the group {current_num})
            # 2. sum of minimums of all non-empty subsets of nums[0...i-1] (these subsets are S')
            #    which is precisely what prefix_min_sum stores.
            min_val_contribution = (current_num + prefix_min_sum) % MOD
            
            # Power term for current_num as the maximum element
            # This is current_num^2 * (current_num + P_i)
            power_term = (pow(current_num, 2, MOD) * min_val_contribution) % MOD
            
            total_power_sum = (total_power_sum + power_term) % MOD
            
            # Update prefix_min_sum for the next iteration (i+1).
            # The recurrence is P_{i+1} = nums[i] + 2 * P_i.
            # Here, current_num is nums[i], and current prefix_min_sum is P_i.
            # So, the next prefix_min_sum will be P_{i+1}.
            prefix_min_sum = (current_num + 2 * prefix_min_sum) % MOD
            
        return total_power_sum