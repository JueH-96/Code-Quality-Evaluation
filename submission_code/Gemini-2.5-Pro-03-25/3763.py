import math
from typing import List

class Solution:
  """
  Finds the minimum y-coordinate H such that the total area of squares above y=H
  equals the total area below y=H.

  The problem requires finding a horizontal line y = H that partitions the total area
  of all squares equally. The total area counts overlapping regions multiple times.
  The minimum such y-coordinate H is sought.

  We can solve this using a sweep-line algorithm. The key insight is that the total area
  below the line y=H, let's call it f(H), is a continuous, piecewise linear, and
  non-decreasing function of H. The slope of f(H) changes only at y-coordinates
  corresponding to the bottom edge (y_i) and top edge (y_i + l_i) of the squares.
  At y=y_i, the slope increases by l_i. At y=y_i+l_i, the slope decreases by l_i.

  The algorithm proceeds as follows:
  1. Calculate the total area of all squares, A_total. The target area is T = A_total / 2.
  2. Create a list of events. Each event corresponds to a y-coordinate where the slope of f(H)
     changes. There are two events per square: (y_i, +1, l_i) indicating the start of contribution
     to the slope, and (y_i + l_i, -1, l_i) indicating the end of contribution.
  3. Sort the events by y-coordinate.
  4. Initialize current accumulated area `current_area = 0`, current slope `current_slope = 0`,
     and the last processed y-coordinate `last_y`. Start `last_y` at the y-coordinate of the first event.
  5. Iterate through the sorted events. For each event point `current_y`:
     a. Calculate the vertical distance covered since the last event: `delta_y = current_y - last_y`.
     b. If `delta_y > 0`, calculate the area accumulated in the interval `(last_y, current_y]`:
        `area_increment = current_slope * delta_y`.
     c. Check if the target area T is reached within this interval: If `current_slope > 0` and
        `current_area < T <= current_area + area_increment`, then the desired H lies in `(last_y, current_y]`.
        Calculate `H = last_y + (T - current_area) / current_slope` and return H. Note the conversion to float is necessary for calculations.
     d. Update `current_area` by adding the `area_increment`. Use integer arithmetic for precision.
     e. Process all events at `current_y`. Update `current_slope` based on the event types (+1 or -1) and lengths `l_i`.
     f. Update `last_y = current_y`.
     g. After processing all events at `current_y`, check if `current_area` is now exactly equal to `T` (within floating point tolerance). If so, return `current_y`. This handles cases where H coincides with an event y-coordinate.

  Using Python's arbitrary precision integers for `current_area` and `current_slope` avoids overflow issues with potentially large coordinates and lengths. Floating point arithmetic is used for `target_area` and the final calculation of H. Standard 64-bit floats (doubles) provide sufficient precision for the required `10^-5` tolerance.
  """
  def separateSquares(self, squares: List[List[int]]) -> float:
    """
    Calculates the minimum separating y-coordinate using a sweep-line algorithm.

    Args:
      squares: A list of squares, where each square is [x_i, y_i, l_i].

    Returns:
      The minimum y-coordinate H as a float.
    """
    total_area = 0
    for _, _, l_i in squares:
        # Calculate total area using integer arithmetic. Python's integers have arbitrary precision.
        total_area += l_i * l_i 
    
    # Constraints state l_i >= 1 and N >= 1, so total_area > 0.
    # No need to check for total_area == 0 explicitly based on constraints.

    target_area = total_area / 2.0 # Target area is half the total area. Use float division.

    events = []
    for _, y_i, l_i in squares:
        # Only squares with positive side length contribute area and affect the slope.
        if l_i > 0: 
            # Start event: At y_i, the square starts contributing to the area below the sweep line.
            # The rate of area increase (slope) increases by l_i.
            events.append((y_i, 1, l_i)) 
            # End event: At y_i + l_i, the square is fully below the sweep line.
            # The rate of area increase (slope) decreases by l_i.
            events.append((y_i + l_i, -1, l_i))

    # If no squares have positive side length, events could be empty.
    # But constraints say l_i >= 1. So events list will contain entries.
    if not events:
         # This case should not happen under problem constraints.
         # If it did, perhaps return 0.0 or handle as an error.
         return 0.0

    events.sort() # Sort events primarily by y-coordinate.

    current_area = 0 # Accumulated area below the sweep line (use Python's arbitrary precision integer)
    current_slope = 0 # Current rate of area accumulation w.r.t y (use Python's arbitrary precision integer)
    
    # Initialize last_y to the y-coordinate of the first event. This correctly handles cases where the minimum y_i > 0.
    last_y = events[0][0]

    idx = 0
    while idx < len(events):
        current_y = events[idx][0] # Get the y-coordinate for the current batch of events
        
        delta_y = current_y - last_y # Vertical distance the sweep line moved since the last event point
        
        # Only perform calculations if the sweep line moved vertically
        if delta_y > 0:
            # Check if the target area is crossed within the interval (last_y, current_y]
            # This check is only relevant if the area is increasing (current_slope > 0).
            if current_slope > 0:
                 # Calculate the area at current_y *before* processing events at current_y.
                 # Use integer arithmetic for precision.
                 area_at_current_y = current_area + current_slope * delta_y
                 
                 # Check if target_area falls within the range [current_area, area_at_current_y].
                 # Comparisons involve float target_area, requires careful conversion.
                 # Check if we were below target and now at or above target.
                 if float(current_area) < target_area and target_area <= float(area_at_current_y): 
                      # Calculate the exact H where area reaches target_area.
                      # The check current_slope > 0 ensures no division by zero.
                      H = float(last_y) + (target_area - float(current_area)) / float(current_slope)
                      # Return H. Added max with float(last_y) as safeguard against potential minor float precision issues causing H < last_y.
                      return max(float(last_y), H) 

            # Update total accumulated area using integer arithmetic based on the slope during the interval.
            current_area += current_slope * delta_y

        # Process all events occurring exactly at current_y coordinate
        event_y_coord = current_y # Store the y-coordinate for this batch of events
        
        while idx < len(events) and events[idx][0] == event_y_coord:
            _, type, l_i = events[idx]
            current_slope += type * l_i # Update slope based on event type (+1 for start, -1 for end)
            idx += 1
        
        # Update last_y to the current event y-coordinate AFTER processing all events at this coordinate
        last_y = event_y_coord

        # After processing events at event_y_coord and updating slope,
        # check if the accumulated area exactly equals the target area.
        # This handles cases where H aligns perfectly with an event y-coordinate.
        # Use a small tolerance (epsilon) for robust floating point comparison.
        if abs(float(current_area) - target_area) < 1e-9: 
             return float(event_y_coord)

    # Fallback return: Under normal circumstances and constraints, this path should not be reached
    # because the function f(H) is continuous and increases from 0 to total_area, so it must cross T.
    # If execution reaches here, it might indicate an edge case not fully handled or precision issues.
    # Returning the last processed y-coordinate might be a reasonable fallback if target T is reached at the very end.
    return float(last_y)