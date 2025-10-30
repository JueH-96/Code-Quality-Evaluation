import math
from typing import List

class Solution:
  def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
    MOD = 10**9 + 7
    n = len(nums)

    if n == 1:
        return 0

    max_val = 0
    min_val = float('inf')
    total_sum = 0
    for x in nums:
        max_val = max(max_val, x)
        min_val = min(min_val, x)
        total_sum += x

    # Case 1: Using two type 1 ops is cheaper or equal to one type 2 op.
    # We can effectively only use type 1 ops.
    # The optimal strategy is to raise all elements to max_val.
    # The case n<=2 also falls here, as op2 is not beneficial over op1.
    if 2 * cost1 <= cost2 or n <= 2:
        cost = (n * max_val - total_sum) * cost1
        return cost % MOD

    # Case 2: 2 * cost1 > cost2 and n > 2. We prefer type 2 operations.
    # Let the final target value be T. T must be >= max_val.
    
    def calculate_cost(T: int) -> int:
        """Calculates the minimum cost to make all elements equal to T."""
        if T < max_val:
            return float('inf')

        # Total increments needed
        S = n * T - total_sum
        # Max increments for a single element
        d_max = T - min_val

        # If the neediest element's requirement is not a bottleneck
        if 2 * d_max <= S:
            # Pair up as many increments as possible.
            cost = (S // 2) * cost2 + (S % 2) * cost1
        else: # The neediest element is the bottleneck
            # Paired operations are limited by increments on other elements.
            ops2 = S - d_max
            # Remaining increments on the neediest element must be individual.
            ops1 = 2 * d_max - S
            cost = ops2 * cost2 + ops1 * cost1
        
        return cost

    # The cost function Cost(T) is piecewise. The minimum occurs either at the
    # boundary (T=max_val) or near the point where the function's form changes.
    # This changeover point, T_switch, is where 2 * d_max(T) ≈ S(T).
    # 2*(T - min_val) = n*T - total_sum => (n-2)*T = total_sum - 2*min_val
    
    min_cost = calculate_cost(max_val)

    A = total_sum - 2 * min_val
    B = n - 2
    
    if A > 0:
        # T_switch is the smallest T where 2*d_max <= S
        T_switch = (A + B - 1) // B  # ceiling division
        
        # We only need to search if T_switch is potentially a better target than max_val.
        # The cost function is increasing for T >= T_switch.
        # So, we check candidates starting from max(max_val, T_switch).
        T_candidate_start = max(max_val, T_switch)
        # A small search window is sufficient due to the nature of the cost function.
        for t_offset in range(4):
            T_cand = T_candidate_start + t_offset
            min_cost = min(min_cost, calculate_cost(T_cand))

    # If A <= 0, the cost function is increasing for all T >= 0.
    # Thus, for T >= max_val, the minimum is at T=max_val.
    
    return min_cost % MOD