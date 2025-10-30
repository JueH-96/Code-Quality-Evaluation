from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        target_subsequences_with_indices = []
        n = len(nums)
        
        def get_target_subsequences_with_indices_recursive(index, current_sum, current_subsequence, current_indices):
            if current_sum == k:
                target_subsequences_with_indices.append((tuple(current_subsequence), tuple(current_indices)))
                return
            if index == n or current_sum > k:
                return
            # Include nums[index]
            get_target_subsequences_with_indices_recursive(index + 1, current_sum + nums[index], current_subsequence + [nums[index]], current_indices + [index])
            # Exclude nums[index]
            get_target_subsequences_with_indices_recursive(index + 1, current_sum, current_subsequence, current_indices)
            
        get_target_subsequences_with_indices_recursive(0, 0, [], [])
        
        total_sum_power = 0
        for subsequence, indices in target_subsequences_with_indices:
            remaining_indices_count = n - len(indices)
            contribution = pow(2, remaining_indices_count, MOD)
            total_sum_power = (total_sum_power + contribution) % MOD
            
        return total_sum_power