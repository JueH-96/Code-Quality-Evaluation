import math
from typing import List

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7

        n = len(nums)
        if n == 0:
            return 0

        max_val = max(nums)
        min_val = min(nums)

        if min_val == max_val:
            return 0

        # Case 1: cost2 is not cheaper than applying cost1 twice
        # If cost2 >= 2 * cost1, it's always optimal or equally optimal to use cost1 for individual increments.
        # This is because increasing two elements with cost2 costs more or equal to increasing them individually with cost1.
        if cost2 >= 2 * cost1:
            total_diff = sum(max_val - x for x in nums)
            return (total_diff * cost1) % MOD
        
        # Case 2: cost2 is cheaper than applying cost1 twice (cost2 < 2 * cost1)
        # We try to maximize the use of cost2 (pair operations).
        # The optimal target value T might be max_val or higher.
        
        sum_nums = sum(nums)
        
        def calculate_cost_for_T(target_T: int) -> int:
            # If target_T is less than max_val, it's an invalid target as we can't decrease elements.
            # (This shouldn't be called with target_T < max_val due to candidate_Ts filtering,
            # but acts as a safeguard and clear logic).
            if target_T < max_val:
                return float('inf') 

            # current_S_T: total sum of increments needed to reach target_T
            current_S_T = n * target_T - sum_nums
            # current_M_T: maximum increments needed for any single element (specifically, the one initially `min_val`)
            current_M_T = target_T - min_val      

            # If current_M_T is 0, it means target_T is equal to min_val.
            # If min_val == max_val (caught at the beginning), cost is 0.
            # If min_val < max_val, but we set target_T = min_val (which is less than max_val), this is invalid.
            # However, the logic for candidate_Ts will ensure target_T >= max_val.
            # So if current_M_T is 0 and target_T >= max_val, it implies min_val == max_val == target_T, thus cost 0.
            if current_M_T == 0:
                return 0 

            cost = 0
            # Condition 1: The element needing `current_M_T` increments needs more than all other increments combined.
            # This is `current_M_T > (current_S_T - current_M_T)`, or `2 * current_M_T > current_S_T`.
            if 2 * current_M_T > current_S_T:
                # `num_pairs` is limited by `current_S_T - current_M_T` (the total increments available from other elements).
                num_pairs = current_S_T - current_M_T
                # The remaining increments for the `min_val` element must be done individually.
                num_singles = current_M_T - num_pairs # Which simplifies to 2 * current_M_T - current_S_T
                cost = num_pairs * cost2 + num_singles * cost1
            else:
                # Condition 2: All increments can be paired up (or one left over if total_S_T is odd).
                # `num_pairs` is simply half of the total increments.
                num_pairs = current_S_T // 2
                num_singles = current_S_T % 2 # 0 if S_T is even, 1 if S_T is odd
                cost = num_pairs * cost2 + num_singles * cost1
            return cost

        min_overall_cost = float('inf')
        
        # Candidate target values for T to check:
        # 1. `max_val`: The initial maximum value in the array. This is the baseline target.
        # 2. `max_val + 1`: Sometimes increasing all elements by just 1 can lead to better pairing options
        #    (e.g., due to parity change in total sum, or shifting the M_T/S_T balance).
        candidate_Ts = {max_val, max_val + 1}
        
        # If n > 2, there might be a critical target value T where the cost function's behavior changes.
        # This occurs when 2 * M_T == S_T. Solving for T: (n-2)T = sum_nums - 2*min_val.
        # So T_boundary_float = (sum_nums - 2*min_val) / (n-2).
        # We need to check integer T values around this floating point boundary.
        if n > 2:
            T_boundary_float = (sum_nums - 2 * min_val) / (n - 2)
            
            # Check integer values around T_boundary_float, ensuring they are at least max_val.
            # A small window (e.g., +/- 2) around floor and ceil is usually sufficient.
            for i in range(-2, 3): 
                # floor(T_boundary_float) + i
                candidate_T1 = max(max_val, math.floor(T_boundary_float) + i)
                candidate_Ts.add(candidate_T1)
                
                # ceil(T_boundary_float) + i
                candidate_T2 = max(max_val, math.ceil(T_boundary_float) + i)
                candidate_Ts.add(candidate_T2)

        # Iterate through all candidate T values and find the minimum cost.
        for t_val in candidate_Ts:
            current_cost = calculate_cost_for_T(t_val)
            min_overall_cost = min(min_overall_cost, current_cost)

        return min_overall_cost % MOD