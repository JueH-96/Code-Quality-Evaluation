import math
from typing import List

class Solution:
    """
    Finds the maximum possible minimum Manhattan distance among k selected points
    lying on the boundary of a square defined by side length 'side'.

    The problem asks to maximize the minimum Manhattan distance between any pair of k selected points.
    This type of "maximize the minimum" problem is often solvable using binary search on the answer.
    We can binary search for the maximum possible minimum distance D.
    For a given candidate distance D, we need a check function `check(D)` that returns True if it's possible
    to select k points such that the Manhattan distance between any two selected points is at least D,
    and False otherwise.

    The check function can be implemented using a greedy strategy. Since the points lie on the boundary
    of a square, we can sort them based on their position along the boundary in counter-clockwise order,
    starting from a point like (0,0). Let the sorted points be P_1, P_2, ..., P_N.

    The greedy strategy works as follows:
    Iterate through each point P_i as a potential starting point for the selection.
    Initialize the selection with P_i. Let the last selected point be P_last = P_i.
    Try to find the next point P_j in the cyclic order after P_last such that the Manhattan distance
    between P_j and P_last is at least D. Select the *first* such P_j encountered. Update P_last = P_j.
    Repeat this process until k points are selected.
    If k points are selected (say P_{s_1}, P_{s_2}, ..., P_{s_k} where P_{s_1} = P_i), perform a final check:
    Is the Manhattan distance between the last selected point P_{s_k} and the starting point P_{s_1} also at least D?
    If yes, then we have found a valid configuration for distance D, and `check(D)` returns True.
    If we cannot find k points for the starting point P_i, or the final check fails, try the next starting point P_{i+1}.
    If no starting point leads to a valid configuration, `check(D)` returns False.

    The time complexity of sorting the points is O(N log N).
    The check function `check(D)` involves iterating through N possible starting points. For each starting point,
    the greedy selection process involves finding k-1 additional points. Finding each subsequent point involves
    scanning potentially all remaining points in cyclic order. However, the scan pointer advances monotonically
    (cyclically). Over the course of finding k-1 points, the total number of points examined by the pointer is O(N).
    Therefore, the work done per starting point is O(N). The total complexity for `check(D)` is O(N^2).
    The binary search performs O(log(side)) iterations.
    The overall time complexity is O(N^2 * log(side)).
    Space complexity is O(N) for storing sorted points.
    """
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        """
        :param side: The edge length of the square.
        :param points: A list of points [x, y] lying on the boundary of the square.
        :param k: The number of points to select.
        :return: The maximum possible minimum Manhattan distance.
        """
        N = len(points)

        # Map points to a 1D representation based on their position along the boundary
        # traversing counter-clockwise starting from (0,0).
        # Using tuples (segment_index, distance_on_segment) ensures correct sorting.
        # Segment indices: 0=bottom, 1=right, 2=top, 3=left.
        def get_1d_coord_tuple(p, side_len):
            x, y = p
            if y == 0: # Bottom edge: (0,0) towards (side,0)
                return (0, x) 
            elif x == side_len: # Right edge: (side,0) towards (side,side)
                 return (1, y)
            elif y == side_len: # Top edge: (side,side) towards (0,side)
                 # distance increases as x decreases
                 return (2, side_len - x)
            elif x == 0: # Left edge: (0,side) towards (0,0)
                 # distance increases as y decreases
                 return (3, side_len - y)
            # Based on problem constraints, points are always on boundary.
            return (-1, -1) # Should not happen

        mapped_points = []
        for i in range(N):
            coord_tuple = get_1d_coord_tuple(points[i], side)
            # Store original point data along with its sorted representation
            mapped_points.append({'coord_tuple': coord_tuple, 'point': points[i]})
        
        # Sort points based on their position along the boundary
        mapped_points.sort(key=lambda p: p['coord_tuple'])
        
        # Extract the sorted list of points for easy access by index 0..N-1
        sorted_points = [p['point'] for p in mapped_points]
        
        # Helper function to compute Manhattan distance between two points given their indices
        # in the sorted_points list.
        def manhattan_dist(p1_idx, p2_idx):
            p1 = sorted_points[p1_idx]
            p2 = sorted_points[p2_idx]
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Check function to determine if it's possible to select k points such that
        # the minimum Manhattan distance between any pair is at least D.
        # Implements the O(N^2) greedy strategy described above.
        def check(D):
            # Trivial case: minimum distance 0 is always possible if k <= N.
            if D == 0: return True 

            # Iterate through each point P_i as a potential starting point for the selection.
            for i in range(N): 
                count = 1 # Number of points selected so far, starting with P_i
                last_idx = i # Index of the last selected point in sorted_points
                
                # Pointer to the next candidate point to check (index in sorted_points)
                # Start searching from the point immediately after P_i in cyclic order.
                current_ptr = (i + 1) % N 
                
                # This loop attempts to find k-1 additional points for the selection.
                for _ in range(k - 1):
                    
                    found_next_in_scan = False # Flag indicating if a suitable next point was found in this scan step
                    
                    # Record the starting position of the scan pointer for this step.
                    # This helps detect if we have scanned the entire circle without finding a suitable point.
                    initial_scan_ptr = current_ptr
                    
                    scan_count = 0 # Counter to ensure scan terminates (at most N points to check)
                    while scan_count < N: 
                        
                        # Calculate Manhattan distance from the *last selected* point to the candidate point
                        dist = manhattan_dist(last_idx, current_ptr)

                        if dist >= D:
                            # Found a suitable point satisfying the minimum distance D
                            last_idx = current_ptr # Update the last selected point index
                            count += 1 # Increment the count of selected points
                             # Update the pointer for the *next* search step to start *after* the current point
                            current_ptr = (current_ptr + 1) % N 
                            found_next_in_scan = True
                            break # Exit the scan (while loop) for this step and proceed to find the next point in the sequence
                        
                        # Point at current_ptr is not suitable, move pointer to the next point cyclically
                        current_ptr = (current_ptr + 1) % N
                        scan_count += 1
                        
                        # Check if we have scanned all N points starting from initial_scan_ptr without finding a point
                        # This condition may be redundant because scan_count < N handles termination,
                        # but explicitly checking helps ensure correctness logic.
                        if current_ptr == initial_scan_ptr:
                             break # Exit scan if full circle checked without finding a point

                    if not found_next_in_scan:
                        # If after scanning (while loop finished or broke due to full circle check), 
                        # no suitable next point was found for this step
                        count = -1 # Mark this starting configuration 'i' as failed
                        break # Exit the loop trying to find k-1 points for this start 'i'

                # After attempting to find k points starting with P_i (completed k-1 steps or failed early)
                if count == k:
                    # If k points were successfully selected using the greedy process
                    # Perform the final check: distance between the last selected point and the starting point P_i.
                    if manhattan_dist(last_idx, i) >= D:
                        # If this distance is also >= D, we found a valid configuration
                        return True 

            # If the loop completes without finding a valid configuration for any starting point 'i'
            return False 

        # Binary search on the possible minimum distance D
        low = 0
        # Maximum possible Manhattan distance is between opposite corners (e.g., (0,0) and (side, side)), which is 2 * side.
        high = 2 * side 
        max_min_dist = 0 # Stores the largest D for which check(D) is True

        while low <= high:
            mid = (low + high) // 2
            
            # Check if it's possible to select k points with minimum distance at least 'mid'
            if check(mid):
                # If check(mid) is true, 'mid' is an achievable minimum distance.
                # Store it as a potential answer and try for a larger distance.
                max_min_dist = mid
                low = mid + 1
            else:
                # If check(mid) is false, 'mid' is too large. Need to try a smaller distance.
                high = mid - 1
                
        # The loop terminates when low > high. max_min_dist holds the maximum D found.
        return max_min_dist