import math
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum cost to make all elements in an array equal
    using two types of operations with associated costs.
    """
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        """
        Calculates the minimum cost to make all elements in the array nums equal.

        Args:
            nums: A list of integers representing the initial array elements.
            cost1: The cost of increasing a single element by 1 (Operation 1).
            cost2: The cost of increasing two distinct elements by 1 each (Operation 2).

        Returns:
            The minimum cost required to make all elements equal, modulo 10^9 + 7.
        """
        MOD = 10**9 + 7  # Define the modulo value.
        
        n = len(nums)
        if n == 0:
            return 0  # Cost is 0 for an empty array.
        
        min_val = min(nums)  # Find the minimum value in the array.
        max_val = max(nums)  # Find the maximum value in the array.
        
        # Calculate the sum of elements. Python's integers handle large numbers automatically.
        sum_nums = sum(nums) 

        # Case 1: Check if using Operation 2 is never more beneficial than using Operation 1 twice.
        # This occurs if 2 * cost1 <= cost2. Also, if n=1, Operation 2 is impossible.
        # In these scenarios, the optimal strategy involves only using Operation 1.
        # The target value to equalize all elements should be the minimum possible, which is max_val.
        if 2 * cost1 <= cost2 or n == 1:
            target = max_val
            # Calculate the total increase needed across all elements to reach max_val.
            total_increase = target * n - sum_nums 
            # The cost is simply the total increase multiplied by cost1. Apply modulo.
            cost = (total_increase * cost1) % MOD
            return cost

        # Case 2: Operation 2 might be cheaper per unit increase (2 * cost1 > cost2 and n > 1).
        # We need to find the optimal target value T (T >= max_val) that minimizes the total cost.
        
        min_total_cost = float('inf')  # Initialize minimum cost found so far to infinity.
        prev_cost = float('inf')       # Initialize the cost of the previous target for optimization check.

        # Iterate through potential target values T. Start from max_val.
        # The cost function Cost(T) usually exhibits unimodal behavior (decreases then increases) near max_val.
        # We explore T values upwards from max_val.
        # A reasonable upper bound for T could be around 2 * max_val. We use 2 * max_val + 3 as a safe limit.
        # The loop includes an optimization to break early if the cost starts increasing.
        for target in range(max_val, 2 * max_val + 3): 
            # Calculate the total increase needed to make all elements equal to the current target T.
            total_increase = target * n - sum_nums
            
            # Calculate the maximum increase required for any single element (the one starting at min_val).
            max_diff = target - min_val 

            # Determine the maximum number of Operation 2 (paired increases) possible, denoted as k2.
            # k2 is limited by two main factors:
            # 1. Half the total increase: k2 <= floor(total_increase / 2). Let this limit be k2_A.
            # 2. The need to balance increases: The element needing max_diff increase imposes a constraint.
            #    This leads to k2 <= total_increase - max_diff. Let this limit be k2_B.
            
            k2_A = total_increase // 2 
            k2_B = total_increase - max_diff 
            
            # k2_B must be non-negative. This holds for target >= max_val because:
            # total_increase - max_diff = (target*n - sum_nums) - (target - min_val)
            # = (n-1)*target - (sum_nums - min_val) >= (n-1)*max_val - (sum_nums - min_val)
            # = sum_{i where nums[i] < max_val} (max_val - nums[i]) >= 0 for n > 1.

            current_cost = 0  # Initialize cost for the current target T.
            
            # The optimal number of Operation 2 applications (k2) is the minimum of the two limits.
            # The remaining increase must be covered by Operation 1 applications (k1).
            if k2_A <= k2_B: 
                 # The total increase amount is the limiting factor. Use k2 = k2_A.
                 k2 = k2_A
                 # k1 = total_increase - 2*k2 which simplifies to total_increase % 2.
                 k1 = total_increase % 2 
                 
                 # Calculate the total cost for this target T.
                 # Python handles large integers, so direct calculation is fine.
                 k1_cost = k1 * cost1
                 k2_cost = k2 * cost2
                 current_cost = k1_cost + k2_cost

            else: # k2_A > k2_B
                 # The balancing constraint related to max_diff is tighter. Use k2 = k2_B.
                 k2 = k2_B
                 # k1 = total_increase - 2*k2 which simplifies to 2*max_diff - total_increase.
                 k1 = total_increase - 2 * k2
                 
                 # Calculate the total cost.
                 k1_cost = k1 * cost1
                 k2_cost = k2 * cost2
                 current_cost = k1_cost + k2_cost

            # Update the overall minimum cost found across all targets evaluated so far.
            min_total_cost = min(min_total_cost, current_cost)
            
            # Optimization: Check if the cost has started increasing.
            # If Cost(T) > Cost(T-1), we assume we've passed the minimum (due to unimodal behavior)
            # and can stop searching further. This check applies from the second target value onwards.
            if target > max_val:
                if current_cost > prev_cost:
                    break  # Stop searching as cost is increasing.
            
            # Store the current cost to compare against in the next iteration.
            prev_cost = current_cost 

        # Return the overall minimum cost found, ensuring the result is modulo MOD.
        return min_total_cost % MOD