from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)

        # The first condition (copy[i] - copy[i-1]) == (original[i] - original[i-1])
        # implies that (copy[i] - original[i]) must be a constant for all i.
        # Let this constant be C. So, copy[i] = original[i] + C.

        # Now we apply the second condition: u_i <= copy[i] <= v_i
        # u_i <= original[i] + C <= v_i
        # This can be broken into two inequalities for each i:
        # 1) u_i <= original[i] + C  =>  u_i - original[i] <= C
        # 2) original[i] + C <= v_i  =>  C <= v_i - original[i]

        # So, for each i, C must be in the range [u_i - original[i], v_i - original[i]].
        # To find a C that works for ALL i, we need to find the intersection of all these ranges.
        # The intersection's lower bound will be the maximum of all individual lower bounds.
        # The intersection's upper bound will be the minimum of all individual upper bounds.

        # Initialize the overall lower and upper bounds for C using the first element's constraints.
        # Since the problem constraints state 2 <= n, original[0] and bounds[0] are guaranteed to exist.
        max_lower_C = bounds[0][0] - original[0]
        min_upper_C = bounds[0][1] - original[0]

        # Iterate through the remaining elements (from the second element onwards)
        for i in range(1, n):
            u_i = bounds[i][0]
            v_i = bounds[i][1]
            orig_i = original[i]

            current_lower_C = u_i - orig_i
            current_upper_C = v_i - orig_i

            # Update the overall max_lower_C (tightest lower bound for C)
            max_lower_C = max(max_lower_C, current_lower_C)
            # Update the overall min_upper_C (tightest upper bound for C)
            min_upper_C = min(min_upper_C, current_upper_C)

        # If the determined overall lower bound for C is greater than the overall upper bound,
        # it means there's no common integer C that satisfies all conditions.
        if max_lower_C > min_upper_C:
            return 0
        
        # Otherwise, the number of possible integer values for C is
        # the count of integers in the inclusive range [max_lower_C, min_upper_C].
        # This count is calculated as (upper_bound - lower_bound + 1).
        return min_upper_C - max_lower_C + 1