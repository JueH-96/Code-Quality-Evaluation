from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        
        # The goal is to assign a unique positive integer height to each tower,
        # respecting its maximumHeight, such that the sum of heights is maximized.

        # Let n be the number of towers. We need to find n distinct positive heights.
        # Let the chosen heights, sorted descending, be h_1 > h_2 > ... > h_n > 0.
        # Let the maximum heights, sorted descending, be m_1 >= m_2 >= ... >= m_n.
        
        # A key insight from combinatorics (related to majorization) is that a valid
        # assignment is possible if and only if h_i <= m_i for all i=1..n.
        # This means the i-th largest chosen height cannot exceed the i-th largest
        # maximum height limit.
        
        # So, the problem reduces to finding a sequence h_1 > ... > h_n > 0
        # that satisfies h_i <= m_i for all i, and maximizes their sum.
        
        # To maximize the sum, we should make each h_i as large as possible.
        # We can determine the optimal h_i values using a greedy approach.
        # For h_1, the max value is m_1. So, we choose h_1 = m_1.
        # For h_2, it must be <= m_2 and < h_1. To maximize it, we choose h_2 = min(m_2, h_1 - 1).
        # In general, for h_i, it must be <= m_i and < h_{i-1}. To maximize it,
        # we choose h_i = min(m_i, h_{i-1} - 1).

        # This logic leads to the following algorithm:
        
        # 1. Sort the maximum heights in descending order to get m_1, m_2, ...
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        
        # This variable will hold the height assigned to the previously considered
        # tower (h_{i-1} in the logic above). For the first tower (i=1), there's
        # no constraint from a previous height, so we can initialize this to a
        # very large value (effectively infinity).
        last_assigned_height = float('inf')
        
        # 2. Iterate through the sorted maximum heights and assign the largest
        #    possible height at each step.
        for m in maximumHeight:
            # Determine the best possible height for the current tower.
            # It's capped by its own maximum `m` and must be less than the
            # height assigned to the tower processed just before it to ensure uniqueness.
            height_to_assign = min(m, last_assigned_height - 1)
            
            # 3. Check for feasibility. All heights must be positive integers.
            if height_to_assign < 1:
                # If the best possible height is 0 or less, we cannot make a
                # valid assignment for this tower (and any subsequent ones).
                # Thus, a valid assignment for all towers is impossible.
                return -1
            
            # 4. If feasible, add the chosen height to the sum and update the
            #    'last_assigned_height' to constrain the next tower's choice.
            total_sum += height_to_assign
            last_assigned_height = height_to_assign
            
        return total_sum