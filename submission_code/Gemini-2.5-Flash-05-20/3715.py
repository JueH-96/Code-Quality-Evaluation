import collections
import bisect
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Step 1: Create a sweep-line event map
        # delta[coord] stores the change in coin density at 'coord'
        # A positive change means a segment starts, a negative change means it ends.
        delta = collections.defaultdict(int)
        for l, r, c in coins:
            delta[l] += c      # At coordinate l, density increases by c
            delta[r + 1] -= c  # At coordinate r+1, density decreases by c
        
        # Step 2: Get all unique coordinates where density changes
        # These are the event points for our sweep line. Sorting them is crucial.
        points = sorted(delta.keys())
        
        # Step 3: Compute prefix sums of coins and densities for intervals
        # prefix_states will store tuples: (coordinate, cumulative_sum_up_to_coord_minus_1, density_at_coord)
        #   - cumulative_sum_up_to_coord_minus_1: Sum of coins from points[0] up to (coordinate - 1).
        #     This effectively means sum for the interval [points[0], coordinate - 1].
        #   - density_at_coord: The constant coin density for the interval [coordinate, next_coordinate - 1].
        prefix_states = []
        current_density = 0  # Represents the coin density for the current interval
        cumulative_sum = 0   # Represents the sum of coins from points[0] up to the current last_coord_processed - 1
        
        # The first coordinate for sum calculation. If segments start at 1, this is 1.
        # If segments start later (e.g., at 100), bags before 100 contain 0 coins.
        # We implicitly assume sum from 1 to points[0]-1 is 0.
        last_coord_processed = points[0] 
        
        for coord in points:
            # Add coins from the interval [last_coord_processed, coord - 1] to the cumulative sum.
            # This interval has coin density 'current_density' (which was established at last_coord_processed).
            cumulative_sum += current_density * (coord - last_coord_processed)
            
            # Update current_density based on the event at 'coord' for the interval [coord, next_coord - 1].
            current_density += delta[coord]
            
            # Store the state: (this_coordinate, sum_up_to_prev_unit, density_from_this_unit_onwards)
            prefix_states.append((coord, cumulative_sum, current_density))
            last_coord_processed = coord
            
        # Step 4: Helper function to query the total coins up to a given coordinate Y
        # F(Y) = sum_{j=1}^{Y} C(j)
        def get_F(Y):
            # If Y is before the first actual event point, the sum is 0 (bags before points[0] have 0 coins).
            if Y < points[0]:
                return 0
            
            # Use bisect_right to find the index `idx` such that prefix_states[idx][0] is the smallest
            # coordinate strictly greater than Y. So, `idx-1` gives the largest coordinate less than or equal to Y.
            # float('inf') is used to make sure the comparison for tuples works correctly for the second and third elements.
            idx = bisect.bisect_right(prefix_states, (Y, float('inf'), float('inf'))) - 1
            
            # Retrieve the state at or just before Y.
            coord_at_idx, F_coord_at_idx, D_coord_at_idx = prefix_states[idx]
            
            # F_coord_at_idx represents the sum of coins from points[0] up to (coord_at_idx - 1).
            # D_coord_at_idx is the constant density for the interval [coord_at_idx, next_point - 1].
            
            # The total sum up to Y is:
            # (Sum from 1 to points[0]-1)  -- this is 0
            # + (Sum from points[0] to coord_at_idx-1) -- this is F_coord_at_idx
            # + (Sum from coord_at_idx to Y) -- this is D_coord_at_idx * (length of interval [coord_at_idx, Y])
            # The length of interval [coord_at_idx, Y] is (Y - coord_at_idx + 1).
            return F_coord_at_idx + D_coord_at_idx * (Y - coord_at_idx + 1)

        # Step 5: Generate candidate starting points for the k-length window
        # The function sum(C(s)...C(s+k-1)) = F(s+k-1) - F(s-1) is piecewise linear.
        # Its maximum occurs at points where its slope changes. These points are
        # when s-1 or s+k-1 coincide with one of the event points (l_i or r_i+1).
        candidate_starts = set()
        for p in points:
            # Candidate 1: Window starts exactly at an event point p.
            candidate_starts.add(p)
            # Candidate 2: Window ends exactly at an event point p. (s + k - 1 = p => s = p - k + 1)
            candidate_starts.add(p - k + 1)
            # Candidate 3: Window starts just after an event point p. (s - 1 = p => s = p + 1)
            candidate_starts.add(p + 1)
            # Candidate 4: Window ends just after an event point p. (s + k - 1 = p + 1 => s = p - k + 2)
            candidate_starts.add(p - k + 2)
        
        # Ensure we also check windows starting from coordinate 1, as it's the first possible bag.
        candidate_starts.add(1) 
        
        # Sort candidates and filter out any starting points less than 1.
        sorted_candidate_starts = sorted([s for s in list(candidate_starts) if s >= 1])
        
        max_coins = 0
        
        # Step 6: Iterate through candidate start points and find the maximum total coins
        for s in sorted_candidate_starts:
            window_end = s + k - 1
            
            # Calculate sum of coins for window [s, window_end] using prefix sums: F(window_end) - F(s-1)
            current_total_coins = get_F(window_end) - get_F(s - 1)
            
            max_coins = max(max_coins, current_total_coins)
            
        return max_coins