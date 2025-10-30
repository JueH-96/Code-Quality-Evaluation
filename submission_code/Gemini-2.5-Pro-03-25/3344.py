import math
from typing import List

class Solution:
    """
    Finds the minimum possible value for the maximum Manhattan distance 
    between any two points after removing exactly one point from the given set.
    
    The Manhattan distance between two points (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.
    
    The maximum Manhattan distance within a set of points P is given by:
      MaxDist(P) = max( max_{p in P}(x+y) - min_{p in P}(x+y),  max_{p in P}(x-y) - min_{p in P}(x-y) )
    Let S = x + y and D = x - y. The formula becomes:
      MaxDist(P) = max( max(S_val) - min(S_val), max(D_val) - min(D_val) )

    When we remove a point p_k, the set becomes P_k = P \ {p_k}. The maximum distance MaxDist(P_k) might decrease.
    We want to find min_{k} MaxDist(P_k).

    Key Insight: The maximum distance is determined by the points with the maximum and minimum S values, and the maximum and minimum D values. Let these points be p_Smax, p_Smin, p_Dmax, p_Dmin.
    If we remove a point p_k that is NOT one of these extremal points (considering potentially multiple points achieving the extrema), the maximum and minimum S and D values remain unchanged. Thus, the maximum distance MaxDist(P_k) remains the same as MaxDist(P).
    To potentially reduce the maximum distance, we must remove one of the points that contribute to the extremal values.
    
    It suffices to consider removing points that are among the top 2 largest or bottom 2 smallest values for S or D. This is because if the point with the absolute maximum/minimum value is removed, the new maximum/minimum will be the second largest/smallest value.
    
    Algorithm:
    1. Compute S = x + y and D = x - y for all points.
    2. Find the indices and values of the points with the top 2 largest S values, bottom 2 smallest S values, top 2 largest D values, and bottom 2 smallest D values. This takes O(N) time.
    3. Collect the unique indices of these critical points (at most 8 points).
    4. For each critical point index k:
        a. Calculate the maximum distance MaxDist(P_k) after removing point p_k. This can be done in O(1) using the precomputed top/bottom 2 values.
        b. Keep track of the minimum MaxDist(P_k) found.
    5. Return the overall minimum max distance.
    
    This approach has an overall time complexity of O(N) because finding the top/bottom 2 elements takes linear time, and checking the small set of critical points takes constant time for each (total O(1)). Space complexity is O(N) to store S and D values.
    """
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Constraint: 3 <= points.length <= 10^5 ensures n >= 3.
        
        # Compute S = x + y and D = x - y for all points
        # Storing tuple (value, index) to keep track of original point index.
        s_values = [(points[i][0] + points[i][1], i) for i in range(n)]
        d_values = [(points[i][0] - points[i][1], i) for i in range(n)]

        # Find indices and values of top 2 maximum S values
        # Initialize with negative infinity to ensure any valid value updates correctly.
        s_max1_val, s_max2_val = -float('inf'), -float('inf')
        s_max1_idx, s_max2_idx = -1, -1
        for i in range(n):
            val = s_values[i][0]
            if val > s_max1_val: # Found a new maximum
                s_max2_val, s_max2_idx = s_max1_val, s_max1_idx # Old max becomes second max
                s_max1_val, s_max1_idx = val, i
            elif val > s_max2_val: # Found a new second maximum
                 s_max2_val, s_max2_idx = val, i

        # Find indices and values of bottom 2 minimum S values
        # Initialize with positive infinity.
        s_min1_val, s_min2_val = float('inf'), float('inf')
        s_min1_idx, s_min2_idx = -1, -1
        for i in range(n):
             val = s_values[i][0]
             if val < s_min1_val: # Found a new minimum
                 s_min2_val, s_min2_idx = s_min1_val, s_min1_idx # Old min becomes second min
                 s_min1_val, s_min1_idx = val, i
             elif val < s_min2_val: # Found a new second minimum
                 s_min2_val, s_min2_idx = val, i

        # Find indices and values of top 2 maximum D values
        d_max1_val, d_max2_val = -float('inf'), -float('inf')
        d_max1_idx, d_max2_idx = -1, -1
        for i in range(n):
            val = d_values[i][0]
            if val > d_max1_val:
                d_max2_val, d_max2_idx = d_max1_val, d_max1_idx
                d_max1_val, d_max1_idx = val, i
            elif val > d_max2_val:
                d_max2_val, d_max2_idx = val, i

        # Find indices and values of bottom 2 minimum D values
        d_min1_val, d_min2_val = float('inf'), float('inf')
        d_min1_idx, d_min2_idx = -1, -1
        for i in range(n):
            val = d_values[i][0]
            if val < d_min1_val:
                d_min2_val, d_min2_idx = d_min1_val, d_min1_idx
                d_min1_val, d_min1_idx = val, i
            elif val < d_min2_val:
                d_min2_val, d_min2_idx = val, i

        # Collect unique critical indices. These are the indices of points that are
        # potentially responsible for the maximum Manhattan distance.
        # We gather indices from top 2 and bottom 2 for both S and D values.
        # Using a set automatically handles duplicate indices.
        critical_indices = {
            s_max1_idx, s_max2_idx, s_min1_idx, s_min2_idx,
            d_max1_idx, d_max2_idx, d_min1_idx, d_min2_idx
        }
        # Remove the placeholder index -1 if it somehow got included (this shouldn't happen for n>=3)
        critical_indices.discard(-1) 

        min_max_dist = float('inf')

        # Iterate through the identified critical points. Removing one of these points
        # is sufficient to find the minimum possible maximum distance.
        for k in critical_indices:
            
            # Calculate max S value in the set P_k = P \ {p_k}
            current_max_s_val = s_max1_val
            if k == s_max1_idx:  # If the point with the maximum S value is removed
                # The new maximum S value becomes the second largest S value.
                current_max_s_val = s_max2_val 
            
            # Calculate min S value in P_k
            current_min_s_val = s_min1_val
            if k == s_min1_idx: # If the point with the minimum S value is removed
                # The new minimum S value becomes the second smallest S value.
                current_min_s_val = s_min2_val 

            # Calculate max D value in P_k
            current_max_d_val = d_max1_val
            if k == d_max1_idx: # If the point with the maximum D value is removed
                # The new maximum D value becomes the second largest D value.
                current_max_d_val = d_max2_val 
                
            # Calculate min D value in P_k
            current_min_d_val = d_min1_val
            if k == d_min1_idx: # If the point with the minimum D value is removed
                # The new minimum D value becomes the second smallest D value.
                current_min_d_val = d_min2_val 
                
            # The maximum Manhattan distance in P_k is the max of the range of S values and D values.
            # Note: Differences are guaranteed non-negative because max >= min.
            max_dist_k = max(current_max_s_val - current_min_s_val, 
                             current_max_d_val - current_min_d_val)
            
            # Keep track of the minimum max distance found across all removals of critical points.
            min_max_dist = min(min_max_dist, max_dist_k)
            
        # Return the overall minimum possible maximum distance.
        # The result will be non-negative.
        return min_max_dist