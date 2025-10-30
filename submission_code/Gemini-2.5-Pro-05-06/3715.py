import bisect

class Solution:
  def maximumCoins(self, coins: List[List[int]], k: int) -> int:
    # Sort coins by start position l_i. This is crucial for get_C.
    coins.sort(key=lambda x: x[0])

    # Store l_i values for quick binary search by get_C.
    starts = [c[0] for c in coins]
    
    # Memoization for get_C. Coordinates can be large, but the number of
    # distinct coordinates we query is limited by roughly 8*N.
    memo_get_C = {}
    def get_C(x_coord: int) -> int:
      if x_coord in memo_get_C:
        return memo_get_C[x_coord]
      
      # Find the rightmost segment whose start l_j is <= x_coord.
      # bisect_right returns an index idx such that all starts[j] <= x_coord for j < idx,
      # and all starts[j] > x_coord for j >= idx.
      # So, the candidate segment is at index idx-1.
      # (coins[idx-1][0] <= x_coord)
      idx = bisect.bisect_right(starts, x_coord)
      
      res = 0
      if idx == 0: # No segment starts at or before x_coord (i.e., x_coord < starts[0])
        res = 0
      else:
        candidate_segment_idx = idx - 1
        l_j, r_j, c_j = coins[candidate_segment_idx]
        
        # Check if x_coord falls within this segment [l_j, r_j]
        if l_j <= x_coord <= r_j:
          res = c_j
        else:
          # x_coord is not in this segment. Since l_j <= x_coord by how candidate_segment_idx was found,
          # it must be that x_coord > r_j.
          res = 0
      
      memo_get_C[x_coord] = res
      return res

    # Collect all event points.
    # Event points are W_start coordinates where C(W_start) or C(W_start + k) might change value.
    # C(x) changes value at segment boundaries l_i or r_i + 1.
    # So, W_start values where C(W_start) might change are l_i and r_i+1.
    # W_start values where C(W_start+k) might change are l_i-k and r_i+1-k (since W_start+k = l_i or W_start+k = r_i+1).
    event_points_set = set()
    if not coins: # Should not happen based on constraints 1 <= coins.length
        return 0
        
    for l, r, c in coins:
      event_points_set.add(l)
      event_points_set.add(r + 1)
      event_points_set.add(l - k)
      event_points_set.add(r + 1 - k)

    # Sort unique event points. These are the W_start values for which we will evaluate S(W_start).
    sorted_event_points = sorted(list(event_points_set))
    
    # Constraints: coins.length >=1, k >= 1.
    # So sorted_event_points will not be empty.
    
    # Calculate S(W_start) for W_start = sorted_event_points[0].
    current_W_start = sorted_event_points[0]
    # The window is [current_W_start, window_end]
    window_end = current_W_start + k - 1 
    
    current_sum = 0
    # Iterate through all coin segments to calculate sum for the first window.
    # For segment [l_seg, r_seg, c_seg], its contribution is:
    # c_seg * length_of_overlap([current_W_start, window_end], [l_seg, r_seg])
    # length_of_overlap(A,B) = max(0, min(A_end, B_end) - max(A_start, B_start) + 1)
    for l_seg, r_seg, c_seg in coins:
        overlap_start = max(l_seg, current_W_start)
        overlap_end = min(r_seg, window_end)
        if overlap_start <= overlap_end: # Check if there is any overlap
            current_sum += c_seg * (overlap_end - overlap_start + 1)
            
    max_so_far = current_sum

    # Sweep the window across the sorted event points.
    # For each interval [sorted_event_points[i], sorted_event_points[i+1]-1] for W_start,
    # the rate of change of S(W_start) is constant.
    for i in range(len(sorted_event_points) - 1):
        prev_W_start = sorted_event_points[i]
        curr_W_start = sorted_event_points[i+1] # This is the W_start for which S() is being computed
        
        dx = curr_W_start - prev_W_start # The distance the window start moved
        
        if dx == 0: # Should not happen if set was used properly, but as a safeguard.
            continue
            
        # S(curr_W_start) = S(prev_W_start) + dx * (C(prev_W_start+k) - C(prev_W_start))
        # C(prev_W_start) is value of coins leaving per unit slide (if dx=1)
        # C(prev_W_start+k) is value of coins entering per unit slide (if dx=1)
        val_leaving_per_step = get_C(prev_W_start)
        val_entering_per_step = get_C(prev_W_start + k)
        
        rate_of_change = val_entering_per_step - val_leaving_per_step
        current_sum += rate_of_change * dx # Update sum to be S(curr_W_start)
        
        max_so_far = max(max_so_far, current_sum)
        
    # Coins c_i are positive, k >= 1. max_so_far will be non-negative.
    return max_so_far