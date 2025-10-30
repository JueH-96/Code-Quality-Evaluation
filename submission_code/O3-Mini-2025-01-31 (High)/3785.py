from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        # copy[0] = x must be chosen such that for each i:
        # u_i <= x + (original[i] - original[0]) <= v_i.
        # Rearranging, we have:
        # x must be in [bounds[i][0] - (original[i] - original[0]), bounds[i][1] - (original[i] - original[0])].
        # The final valid x must lie in the intersection of all these intervals.
        
        x_low = bounds[0][0]  # for i == 0, it's exactly the bounds for x
        x_high = bounds[0][1]
        
        base = original[0]
        for i in range(1, n):
            diff = original[i] - base
            # The condition for index i is:
            # x + diff in [bounds[i][0], bounds[i][1]]
            # Hence, x must lie in:
            # [bounds[i][0] - diff, bounds[i][1] - diff]
            curr_low = bounds[i][0] - diff
            curr_high = bounds[i][1] - diff
            
            # Update the global intersection
            x_low = max(x_low, curr_low)
            x_high = min(x_high, curr_high)
            
            # If there is no intersection, no valid array exists
            if x_low > x_high:
                return 0
        
        # The number of valid arrays is the number of valid x.
        return x_high - x_low + 1