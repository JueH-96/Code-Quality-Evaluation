from typing import List
import collections

class Solution:
  def maximumBeauty(self, nums: List[int], k: int) -> int:
    # For each number nums[i], it can be changed to any integer in the range
    # [nums[i] - k, nums[i] + k]. Let this be interval I_i = [L_i, R_i].
    # We want to find an integer X such that X is covered by the maximum number of intervals I_i.
    # This is a classic sweep-line algorithm problem.
    # An interval [L, R] (inclusive) contributes +1 to the count of overlapping intervals
    # for all points from L to R.
    # In the sweep-line approach, this means an event `(+1)` at point `L`
    # and an event `(-1)` at point `R+1`.

    # We use a dictionary (or Counter/defaultdict) to store the net change
    # in the number of active intervals at each event point.
    # points_delta[coord] will store the sum of +1s for intervals starting at `coord`
    # and -1s for intervals ending at `coord-1` (i.e., whose R_i was `coord-1`).
    
    # Using collections.defaultdict(int) for convenience.
    # A standard dict with `dict.get(key, 0) + value` would also work.
    # A collections.Counter is also suitable.
    points_delta = collections.defaultdict(int)

    for num_val in nums:
      start_interval = num_val - k
      end_interval = num_val + k
      
      points_delta[start_interval] += 1
      # The interval [L, R] ends at R. So, it's no longer active starting from R+1.
      # Thus, the decrement event is at R+1.
      points_delta[end_interval + 1] -= 1
      
    # Get all distinct event coordinates and sort them.
    # These are the points where the number of overlapping intervals *might* change.
    sorted_coords = sorted(points_delta.keys())
    
    current_active_intervals = 0
    max_active_intervals = 0
    
    # Iterate through the sorted unique coordinates.
    # The `current_active_intervals` count calculated after processing `coord`
    # (i.e., after adding points_delta[coord]) represents the number of intervals
    # that cover the point `coord` itself. This count is valid for all integers X
    # in the range `[coord, next_coord - 1]` (where `next_coord` is the
    # subsequent coordinate in `sorted_coords`).
    for coord in sorted_coords:
      # Apply the net change associated with this coordinate.
      current_active_intervals += points_delta[coord]
      # After updating, `current_active_intervals` is the number of intervals
      # covering the point `coord` (and subsequent points until the next event).
      # We update `max_active_intervals` with this value.
      max_active_intervals = max(max_active_intervals, current_active_intervals)
      
    return max_active_intervals