class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        def get_subsequences_sum_k(current_nums, target_sum):
            result_subsequences = []
            
            def find_subsequences(index, current_sum, current_subsequence):
                if current_sum == target_sum:
                    result_subsequences.append(tuple(current_subsequence))
                    return
                if current_sum > target_sum or index == len(current_nums):
                    return
                
                # Include nums[index]
                find_subsequences(index + 1, current_sum + current_nums[index], current_subsequence + [current_nums[index]])
                # Exclude nums[index]
                find_subsequences(index + 1, current_sum, current_subsequence)
                
            find_subsequences(0, 0, [])
            return list(set(result_subsequences))

        subsequences_sum_k = get_subsequences_sum_k(nums, k)
        total_power_sum = 0
        for subsequence in subsequences_sum_k:
            length_T = len(subsequence)
            contribution = pow(2, n - length_T, MOD)
            total_power_sum = (total_power_sum + contribution) % MOD
            
        return total_power_sum

from typing import List