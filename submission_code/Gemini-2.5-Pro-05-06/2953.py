import collections
from typing import List

class Solution:
  def countPairs(self, coordinates: List[List[int]], k: int) -> int:
    ans = 0
    
    # The 'counts' Counter will store the frequency of each point (x, y)
    # encountered so far in the iteration.
    # Using collections.Counter is convenient because it defaults to 0 for missing keys.
    counts = collections.Counter()

    # Iterate through each point P_j = (x_j, y_j) in the input list 'coordinates'.
    # In this loop, (x_j, y_j) represents the current point being processed.
    for x_j, y_j in coordinates:
        # For the current point (x_j, y_j), we want to find the number of
        # previously processed points P_i = (x_i, y_i) (where index i < index j)
        # such that the XOR distance is k.
        # The XOR distance is (x_i XOR x_j) + (y_i XOR y_j).
        
        # Let x_xor_component = x_i XOR x_j
        # Let y_xor_component = y_i XOR y_j
        # We need x_xor_component + y_xor_component = k.
        
        # We can iterate through all possible values for x_xor_component.
        # Since x_xor_component must be non-negative, it can range from 0 to k.
        # Let `val1` be a candidate for x_xor_component.
        for val1 in range(k + 1): # val1 iterates from 0 to k (inclusive)
            # If x_xor_component is val1, then y_xor_component must be k - val1.
            # Let `val2` be this required value for y_xor_component.
            val2 = k - val1
            
            # Now, we need to find x_i and y_i (coordinates of a previous point P_i) such that:
            # 1. x_i XOR x_j = val1  =>  x_i = x_j XOR val1 (properties of XOR)
            # 2. y_i XOR y_j = val2  =>  y_i = y_j XOR val2
            
            target_x_i = x_j ^ val1
            target_y_i = y_j ^ val2
            
            # The 'counts' map stores frequencies of points (x,y) that appeared
            # *before* the current point (x_j, y_j).
            # So, counts[(target_x_i, target_y_i)] gives the number of
            # suitable P_i points that form a valid pair with P_j.
            ans += counts[(target_x_i, target_y_i)]
        
        # After processing (x_j, y_j) against all previous points,
        # add (x_j, y_j) to the 'counts' map. This makes it available
        # for subsequent points to form pairs with.
        counts[(x_j, y_j)] += 1
            
    return ans