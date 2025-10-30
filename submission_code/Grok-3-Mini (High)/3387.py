from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        num_leq_needed = (n // 2) + 1
        num_geq_needed = n - (n // 2)
        
        # Extract lists of numbers greater than k and less than k
        greater_than_k_list = [num for num in nums if num > k]
        less_than_k_list = [num for num in nums if num < k]
        
        # Current number of elements <= k and >= k
        L = n - len(greater_than_k_list)  # num <= k
        G = n - len(less_than_k_list)     # num >= k
        
        # Calculate the deficiencies
        delta_L = max(0, num_leq_needed - L)
        delta_G = max(0, num_geq_needed - G)
        
        cost = 0
        
        # Cost to make enough elements <= k
        if delta_L > 0:
            sorted_greater = sorted(greater_than_k_list)
            cost += sum(sorted_greater[:delta_L]) - delta_L * k
        
        # Cost to make enough elements >= k
        if delta_G > 0:
            sorted_less = sorted(less_than_k_list)
            cost += delta_G * k - sum(sorted_less[-delta_G:])
        
        return cost