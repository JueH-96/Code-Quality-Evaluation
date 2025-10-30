import math
from typing import List

class Solution:
  def minimumDistance(self, points: List[List[int]]) -> int:
    n = len(points)

    # Per constraints, n >= 3.
    # If n were <= 2, the max distance after removing one point would be 0.
    # For n=3, removing one point leaves 2 points. The max distance is simply
    # the distance between these two remaining points. The logic handles this:
    # e.g., for n=3, s_vals has indices 0, 1, 2.
    # s_vals[0] is min, s_vals[1] is middle, s_vals[2] (which is s_vals[n-1]) is max.
    # s_vals[n-2] would be s_vals[1]. This is correct.

    # Stores (value, original_index) to keep track of which point corresponds to which value
    s_vals = []  # For x+y sums
    d_vals = []  # For x-y differences

    for i in range(n):
      x, y = points[i]
      s_vals.append((x + y, i))
      d_vals.append((x - y, i))

    # Sort based on s-values (x+y) and d-values (x-y) respectively.
    # Python's default sort on tuples sorts by the first element, then by the second if first elements are tied.
    # This behavior is fine. We only care that s_vals[0] contains *a* point with the minimum sum,
    # s_vals[1] contains *a* point with the second minimum sum (which might be equal to the first), etc.
    s_vals.sort()
    d_vals.sort()

    min_overall_max_dist = math.inf # Initialize with a very large number

    # Iterate through each point by its original index.
    # Consider removing points[i_removed_orig_idx] from the set.
    for i_removed_orig_idx in range(n):
      # For the remaining n-1 points, find the maximum Manhattan distance.
      # This distance is max( (max s-value - min s-value), (max d-value - min d-value) )
      # among the remaining points.
      
      # Variables to store the new min/max s-values and d-values after removal
      current_s_min_val = 0
      current_s_max_val = 0
      current_d_min_val = 0
      current_d_max_val = 0

      # Determine S'_min (new minimum s-value after removing points[i_removed_orig_idx])
      # s_vals[0] is (value, original_index_of_point_with_min_s_val)
      # s_vals[1] is (value, original_index_of_point_with_second_min_s_val)
      if s_vals[0][1] == i_removed_orig_idx:
        # If the point that originally had the smallest s-value is the one being removed,
        # the new minimum s-value will be from the point that originally had the second smallest s-value.
        current_s_min_val = s_vals[1][0]
      else:
        # Otherwise, the point with the smallest s-value is still in the set, so S_min is unchanged.
        current_s_min_val = s_vals[0][0]
      
      # Determine S'_max (new maximum s-value)
      # s_vals[n-1] is (value, original_index_of_point_with_max_s_val)
      # s_vals[n-2] is (value, original_index_of_point_with_second_max_s_val)
      if s_vals[n-1][1] == i_removed_orig_idx:
        # If the point with the overall maximum s-value is removed,
        # the new maximum s-value comes from the point that had the second largest s-value.
        current_s_max_val = s_vals[n-2][0]
      else:
        # Otherwise, the maximum s-value is unchanged.
        current_s_max_val = s_vals[n-1][0]

      # Determine D'_min (new minimum d-value)
      if d_vals[0][1] == i_removed_orig_idx:
        current_d_min_val = d_vals[1][0]
      else:
        current_d_min_val = d_vals[0][0]

      # Determine D'_max (new maximum d-value)
      if d_vals[n-1][1] == i_removed_orig_idx:
        current_d_max_val = d_vals[n-2][0]
      else:
        current_d_max_val = d_vals[n-1][0]
      
      # Calculate the maximum Manhattan distance for the current set of (n-1) points
      max_dist_for_this_removal = max(current_s_max_val - current_s_min_val, 
                                      current_d_max_val - current_d_min_val)
      
      # Update the overall minimum of these maximum distances
      if max_dist_for_this_removal < min_overall_max_dist:
        min_overall_max_dist = max_dist_for_this_removal
        
    return min_overall_max_dist