import heapq
from typing import List

class Solution:
  def minDifference(self, nums: List[int]) -> int:
    n = len(nums)

    # Max coordinate value any x or y can take.
    # Max existing num is 10^9. Max k is 10^9. So fill value can be 10^9+10^9 = 2*10^9.
    # Min positive fill value is 1.
    # Using a slightly larger constant for r_j+1 calculations to avoid issues with exact boundary.
    MAX_COORD_VAL = 2 * 10**9 + 7 

    def check(k: int) -> bool:
        # Step 1: Check fixed pairs
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                if abs(nums[i] - nums[i+1]) > k:
                    return False
        
        # Step 2: Construct intervals P = {[l_j, r_j]} for each nums[j] == -1
        potential_intervals = []
        for j in range(n):
            if nums[j] == -1:
                l_j, r_j = 1, MAX_COORD_VAL # Initial full range for a positive integer
                
                # Constraint from left neighbor
                if j > 0 and nums[j-1] != -1:
                    l_j = max(l_j, nums[j-1] - k)
                    r_j = min(r_j, nums[j-1] + k)
                
                # Constraint from right neighbor
                if j < n - 1 and nums[j+1] != -1:
                    l_j = max(l_j, nums[j+1] - k)
                    r_j = min(r_j, nums[j+1] + k)
                
                l_j = max(1, l_j) # Ensure positive, as x and y must be positive
                
                if l_j > r_j: # This -1 cannot be filled satisfying constraints
                    return False
                # Store interval along with its original index, though original_idx isn't used later
                potential_intervals.append({'l': l_j, 'r': r_j, 'original_idx': j})

        if not potential_intervals: # No -1s, and fixed parts are okay (from step 1)
            return True

        # Step 3: Check if x=y works (i.e., one value can fill all -1s)
        glob_L, glob_R = 1, MAX_COORD_VAL # Range for a single fill value
        for p_interval in potential_intervals:
            glob_L = max(glob_L, p_interval['l'])
            glob_R = min(glob_R, p_interval['r'])
        
        if glob_L <= glob_R:
            # We can pick x = y = glob_L. This satisfies |x-y| <= k (as 0 <= k).
            # And x=y=glob_L is in every p_interval's required range [l_j, r_j].
            return True

        # Step 4: Sweep line for x. Check if two values x, y can work.
        # Event points: (coordinate, type, l_val, r_val, interval_system_idx)
        # type: 0 for START of interval coverage by x, 1 for END
        # interval_system_idx is the index in potential_intervals list, used for heap item uniqueness.
        events = []
        for i_idx, p_interval in enumerate(potential_intervals):
            events.append((p_interval['l'], 0, p_interval['l'], p_interval['r'], i_idx)) 
            # r_j+1 because x covering [l_j, r_j] means x can be r_j.
            # If x becomes r_j+1, it no longer covers [l_j, r_j].
            events.append((p_interval['r'] + 1, 1, p_interval['l'], p_interval['r'], i_idx)) 
        
        events.sort() # Sort events by coordinate, then by type (START before END implicitly if types are 0 and 1)

        # Heaps for L_y (max of l_j for inactive intervals)
        inactive_l_heap = [] # stores (-l_j, interval_system_idx) for max-heap behavior
        inactive_l_removed_heap = [] # stores removed items for lazy deletion

        # Heaps for R_y (min of r_j for inactive intervals)
        inactive_r_heap = [] # stores (r_j, interval_system_idx)
        inactive_r_removed_heap = [] # stores removed items for lazy deletion

        # Initially, all intervals are "inactive" (not covered by current sweep-line x position concept).
        # These heaps will store l_j/r_j for intervals NOT covered by x.
        for i_idx, p_interval in enumerate(potential_intervals):
            heapq.heappush(inactive_l_heap, (-p_interval['l'], i_idx))
            heapq.heappush(inactive_r_heap, (p_interval['r'], i_idx))
        
        for i in range(len(events)):
            coord, type, l_val, r_val, interval_system_idx = events[i]
            
            if type == 0: # START event: x now covers this interval. Remove from inactive sets.
                heapq.heappush(inactive_l_removed_heap, (-l_val, interval_system_idx))
                heapq.heappush(inactive_r_removed_heap, (r_val, interval_system_idx))
            else: # END event: x no longer covers this interval. Add to inactive sets.
                heapq.heappush(inactive_l_heap, (-l_val, interval_system_idx))
                heapq.heappush(inactive_r_heap, (r_val, interval_system_idx))

            # After processing event(s) at `coord`, check candidate range for x.
            # If next event is at the same coordinate, process it first.
            if i + 1 < len(events) and events[i+1][0] == coord:
                continue

            # Current range for x to be chosen from: [coord, next_event_coord - 1]
            x_test_start = coord
            x_test_end = MAX_COORD_VAL # Default if this is the last event coord
            if i + 1 < len(events):
                x_test_end = events[i+1][0] - 1
            
            if x_test_start > x_test_end: # Empty interval for x candidate values
                continue
            
            # Clean heaps to get actual current max L_y and min R_y for inactive intervals
            while inactive_l_removed_heap and inactive_l_heap and inactive_l_heap[0] == inactive_l_removed_heap[0]:
                heapq.heappop(inactive_l_heap)
                heapq.heappop(inactive_l_removed_heap)
            while inactive_r_removed_heap and inactive_r_heap and inactive_r_heap[0] == inactive_r_removed_heap[0]:
                heapq.heappop(inactive_r_heap)
                heapq.heappop(inactive_r_removed_heap)

            if not inactive_l_heap: # All intervals are covered by x
                # We can pick y = x. This is always possible if k >=0.
                # Since x_test_start <= x_test_end, there is a valid choice for x.
                return True
            else:
                # L_y is max l_j among intervals not covered by x
                # R_y is min r_j among intervals not covered by x
                L_y = -inactive_l_heap[0][0] 
                R_y = inactive_r_heap[0][0]  

                if L_y > R_y: # Inconsistent requirements for y from non-covered intervals
                    continue # This range of x_test values won't work.

                # We need to find an x_try in [x_test_start, x_test_end]
                # and a y_try such that y_try \in [max(1, x_try-k), x_try+k] (condition |x_try-y_try|<=k and y_try positive)
                # and y_try \in [L_y, R_y] (y covers remaining intervals).
                # This is possible if the range for x_try derived from y's constraints,
                # [max(1, L_y-k), R_y+k], intersects with x's current candidate range [x_test_start, x_test_end].
                
                target_x_range_min = max(1, L_y - k)
                target_x_range_max = R_y + k
                
                # Find intersection of [x_test_start, x_test_end] and [target_x_range_min, target_x_range_max]
                intersect_L = max(x_test_start, target_x_range_min)
                intersect_R = min(x_test_end, target_x_range_max)

                if intersect_L <= intersect_R: # If intersection is non-empty
                    return True
        
        return False

    # Binary search for k
    low = 0
    high = 10**9 # Max possible difference for nums values in [1, 10^9]
    ans = high

    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    
    return ans