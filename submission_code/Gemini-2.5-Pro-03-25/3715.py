import collections
import bisect
from typing import List

class Solution:
  """
  Finds the maximum number of coins obtainable by collecting k consecutive bags.
  Uses an event point based prefix sum approach optimized for sparse updates.
  The coordinates and k can be large, requiring efficient computation without iterating through all bags.
  """
  def maximumCoins(self, coins: List[List[int]], k: int) -> int:
    """
    Calculates the maximum coins using prefix sums computed at event points.

    Args:
      coins: A list of segments [l_i, r_i, c_i], where bags from l_i to r_i each contain c_i coins.
             Segments are non-overlapping.
      k: The number of consecutive bags to collect.

    Returns:
      The maximum number of coins possible by collecting k consecutive bags.
    """

    # Use a dictionary to store changes in coin counts at specific positions.
    # For a segment [l, r, c], the number of coins per bag increases by c at position l, 
    # and decreases by c at position r+1 (since the segment includes r).
    events = collections.defaultdict(int)
    has_coins = False
    for l, r, c in coins:
      # Skip segments with 0 coins, they don't affect sums or event points.
      if c == 0: continue 
      has_coins = True
      events[l] += c
      events[r + 1] -= c
        
    # If no segments have positive coins, the maximum obtainable coins is 0.
    if not has_coins: 
        return 0

    # Get sorted unique positive positions where coin counts change.
    # Constraints state l >= 1, so event positions l will be >= 1. 
    # r+1 will be >= 2. So all event positions should be positive.
    positions = sorted(events.keys())
    
    # Filter out potentially non-positive positions if constraints were weaker. 
    # In this problem, this filtering might be redundant but safe.
    positions = [p for p in positions if p > 0]

    if not positions: 
         # This case should not happen given constraints unless all c=0.
         return 0

    # prefix_sum_at_boundary maps `pos-1` to the prefix sum PS(pos-1) = sum_{y=1 to pos-1} C(y).
    # This map stores prefix sums calculated just before each event point `pos`.
    # Key 0 maps to PS(0)=0.
    prefix_sum_at_boundary = {0: 0} 
    
    # interval_values maps an event position `pos` to the coin value C(x) for x in the interval starting at `pos`.
    # interval_values[pos] = C_pos, where C_pos is value for bags in [pos, next_pos-1].
    interval_values = {} 
    
    current_coin_value = 0  # Tracks the number of coins per bag in the current interval.
    last_pos = 0 # Track the previous event position processed. Start conceptually at 0.

    # Compute prefix sums at boundaries (points P_j - 1) and interval coin values (C_j).
    for pos in positions:
        # Calculate the contribution of the interval [last_pos, pos-1] to the prefix sum.
        if pos > last_pos:
            interval_len = pos - last_pos
            # Get the prefix sum up to the start of the interval (last_pos - 1).
            # Use .get() for safety, especially for the initial state (last_pos=0, needs PS(-1) which is 0).
            ps_at_last_pos_minus_1 = prefix_sum_at_boundary.get(last_pos - 1, 0) 
            
            # Calculate the total coins in the interval [last_pos, pos-1] using the coin value *before* the current event at `pos`.
            current_interval_sum = current_coin_value * interval_len
            
            # The new prefix sum up to pos-1 is PS(last_pos-1) + sum over [last_pos, pos-1].
            new_ps = ps_at_last_pos_minus_1 + current_interval_sum
            prefix_sum_at_boundary[pos - 1] = new_ps
           
        # Apply the change in coin value at the current event position `pos`.
        change = events[pos]
        current_coin_value += change
        # Store the new coin value effective for the interval starting *at* pos.
        interval_values[pos] = current_coin_value 
        
        # Update last processed position.
        last_pos = pos

    # Store the sorted unique positive event positions for binary search access.
    P = positions 
    M = len(P)

    # Helper function to compute prefix sum PS(x) = sum_{y=1 to x} C(y) efficiently.
    def get_prefix_sum(x):
        if x <= 0:
            return 0
        
        # Find the index `idx` in the sorted positions list `P` such that P[idx-1] <= x < P[idx].
        # `bisect_right` finds the insertion point for x. All P[i] <= x are to the left of idx.
        idx = bisect.bisect_right(P, x)
        
        if idx == 0:
             # x is smaller than the smallest event position P[0].
             # By problem setup, C(y)=0 for 1 <= y < P[0]. So PS(x) = 0.
             return 0

        # The interval relevant for x starts at P_j = P[idx-1].
        j = idx - 1
        P_j = P[j]
        
        # Coin value C_j in the interval starting at P_j is C(x) for P_j <= x < P_{j+1}.
        # This value was stored in interval_values.
        C_j = interval_values[P_j]
        
        # We need prefix sum up to P_j - 1, which is PS(P_j - 1).
        # This value is stored in prefix_sum_at_boundary map with key P_j - 1.
        # Use .get() with default 0 for safety. It correctly handles P_j=1 (needs PS(0)=0) 
        # and cases where the interval preceding P_j had 0 value and wasn't explicitly stored.
        ps_at_Pj_minus_1 = prefix_sum_at_boundary.get(P_j - 1, 0) 

        # Compute PS(x) using the formula: PS(x) = PS(P_j - 1) + C_j * (number of elements from P_j to x).
        # Number of elements in [P_j, x] is (x - P_j + 1).
        ps_x = ps_at_Pj_minus_1 + C_j * (x - P_j + 1)
        
        return ps_x

    max_coins = 0
    
    # Determine the set of critical starting positions `p` to check.
    # The maximum sum S(p) = sum_{x=p}^{p+k-1} C(x) can occur when the window start `p` or end `p+k-1` 
    # aligns with points where C(x) changes value. These points are the event points `e`.
    # The rate of change of S(p), S(p+1)-S(p) = C(p+k) - C(p), changes when `p` or `p+k` hits an event point `e`.
    # This happens when p = e or p = e - k.
    # Critical positions are {e} U {e-k} for all event points e = l_i or e = r_i + 1.
    critical_positions = set()
    all_event_points = set(events.keys()) # Using the raw keys from events map

    for e in all_event_points:
         # Candidate starting position p = e
         if e >= 1: critical_positions.add(e)
         # Candidate starting position p = e - k
         p_e_minus_k = e - k
         if p_e_minus_k >= 1: critical_positions.add(p_e_minus_k)
         
    # Add position 1 as a potential starting point, it's the first valid bag index.
    # This ensures we consider windows starting at the very beginning.
    critical_positions.add(1) 

    # Iterate through critical starting positions and calculate the sum for the window [p, p+k-1].
    for p in critical_positions:
        # Window covers bags from p to p+k-1 inclusive.
        window_start = p
        window_end = p + k - 1
        
        # Calculate sum in window using the efficient prefix sum function.
        # Sum = PS(window_end) - PS(window_start - 1)
        current_sum = get_prefix_sum(window_end) - get_prefix_sum(window_start - 1)
        max_coins = max(max_coins, current_sum)
        
    return max_coins