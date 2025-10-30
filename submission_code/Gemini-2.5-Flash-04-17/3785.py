from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        """
        Counts the number of possible arrays 'copy' satisfying the given conditions.

        Args:
            original: The original array of length n.
            bounds: The 2D array of shape n x 2, where bounds[i] = [u_i, v_i].

        Returns:
            The number of possible arrays 'copy'.
        """
        n = len(original)
        
        # The condition copy[i] - copy[i-1] == original[i] - original[i-1] for i >= 1
        # implies that the difference copy[i] - original[i] is constant for all i >= 0.
        # This can be shown by rearranging the equation:
        # copy[i] - original[i] = copy[i-1] - original[i-1].
        # Let this constant difference be C. So, copy[i] = original[i] + C for all i.
        
        # The second condition is bounds[i][0] <= copy[i] <= bounds[i][1] for 0 <= i <= n-1.
        # Substituting copy[i] = original[i] + C into the bounds inequality:
        # bounds[i][0] <= original[i] + C <= bounds[i][1].
        
        # Rearranging the inequality to isolate C:
        # bounds[i][0] - original[i] <= C <= bounds[i][1] - original[i].
        
        # This inequality must hold true for every index i from 0 to n-1.
        # Therefore, C must be an integer that simultaneously satisfies all these inequalities.
        # This means C must be greater than or equal to the maximum of all the lower bounds (bounds[i][0] - original[i])
        # and less than or equal to the minimum of all the upper bounds (bounds[i][1] - original[i]).
        
        # Let min_c_overall be the maximum of all lower bounds for C derived from each index i.
        # min_c_overall = max_{0 <= i <= n-1}(bounds[i][0] - original[i])
        
        # Let max_c_overall be the minimum of all upper bounds for C derived from each index i.
        # max_c_overall = min_{0 <= i <= n-1}(bounds[i][1] - original[i])
        
        # We can compute these overall bounds by iterating through the indices.
        # Initialize min_c_overall and max_c_overall using the constraints for the first element (i=0).
        # The bounds for C from i=0 are: bounds[0][0] - original[0] <= C <= bounds[0][1] - original[0].
        min_c_overall = bounds[0][0] - original[0]
        max_c_overall = bounds[0][1] - original[0]
        
        # Iterate through the remaining elements from index 1 to n-1.
        # For each index i, calculate the bounds it imposes on C and update the overall bounds.
        for i in range(1, n):
            # The constraints for C from index i are:
            # bounds[i][0] - original[i] <= C <= bounds[i][1] - original[i]
            lower_bound_c_i = bounds[i][0] - original[i]
            upper_bound_c_i = bounds[i][1] - original[i]
            
            # To satisfy all constraints, the possible range for C must be the intersection
            # of the current overall range [min_c_overall, max_c_overall] and the new range
            # [lower_bound_c_i, upper_bound_c_i].
            # The new overall minimum lower bound is the maximum of the current minimum and the new lower bound.
            min_c_overall = max(min_c_overall, lower_bound_c_i)
            
            # The new overall maximum upper bound is the minimum of the current maximum and the new upper bound.
            max_c_overall = min(max_c_overall, upper_bound_c_i)
        
        # After iterating through all indices, the possible integer values for C are in the range [min_c_overall, max_c_overall].
        # The number of integers in this range is max_c_overall - min_c_overall + 1, provided that min_c_overall <= max_c_overall.
        # If min_c_overall > max_c_overall, it means there is no integer C that satisfies all constraints,
        # so the range is empty and the number of possible arrays is 0.
        
        # The difference max_c_overall - min_c_overall + 1 correctly gives the count if the range is valid.
        # If the range is invalid (max_c_overall < min_c_overall), the difference will be negative or zero.
        # We use max(0, ...) to handle the case where the range is invalid and ensure the result is non-negative.
        
        num_possible_c = max_c_overall - min_c_overall + 1
        
        return max(0, num_possible_c)