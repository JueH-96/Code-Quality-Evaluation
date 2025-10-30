import bisect
from typing import List

class Solution:
    """
    Determines if it's possible to make two horizontal or two vertical cuts on an NxN grid 
    such that each of the three resulting sections contains at least one rectangle,
    and every rectangle belongs to exactly one section. The cuts must be made at integer coordinates
    strictly between 0 and n. A cut at coordinate `c` is valid if no rectangle crosses it, i.e., 
    for any rectangle with interval [start, end] in the dimension of the cut, it's not the case that start < c < end.
    """
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        
        # Function to compute the union of open intervals (start, end)
        def compute_union_open_intervals(intervals: List[List[int]]) -> List[List[int]]:
            """
            Computes the union of a list of open intervals (start, end).
            Input intervals are [start, end]. The function processes them as open intervals (start, end).
            Returns a list of disjoint intervals [s, e] representing the union of open intervals (s, e).
            """
            intervals_to_merge = []
            for start, end in intervals:
                 # We only care about intervals where start < end, as these define non-empty open intervals.
                 if start < end: 
                     intervals_to_merge.append([start, end])
            
            if not intervals_to_merge: return []
            
            # Sort intervals based on their start coordinate.
            intervals_to_merge.sort() 
            
            merged = []
            if not intervals_to_merge: return [] # Check again after filtering empty intervals

            # Initialize the first merged interval.
            current_start, current_end = intervals_to_merge[0]
            
            # Iterate through the sorted intervals to merge overlapping ones.
            for i in range(1, len(intervals_to_merge)):
                next_start, next_end = intervals_to_merge[i]
                
                # Check for overlap considering open intervals: (current_start, current_end) and (next_start, next_end)
                # Overlap occurs if the start of the next interval is before the end of the current merged interval.
                if next_start < current_end: 
                    # Merge by extending the end of the current interval to the maximum end coordinate.
                    current_end = max(current_end, next_end) 
                else: 
                    # No overlap, finalize the current merged interval and start a new one.
                    merged.append([current_start, current_end])
                    current_start, current_end = next_start, next_end
            
            # Add the last merged interval after the loop finishes.
            merged.append([current_start, current_end]) 
            return merged

        # Function to check if integer cut 'c' is invalid (falls within the union of open intervals)
        def check_invalid(c: int, union_intervals: List[List[int]]) -> bool:
            """
            Checks if an integer 'c' falls strictly inside any interval (s, e) in the union.
            union_intervals is a sorted list of disjoint [s, e] representing open intervals (s, e).
            Uses binary search to efficiently find the relevant interval.
            """
            # Perform binary search to find the interval potentially containing c.
            # Find the index `idx` such that union_intervals[idx][0] is the first start coordinate >= c.
            low, high = 0, len(union_intervals)
            found_interval_idx = -1 
            while low < high:
                 mid = (low + high) // 2
                 if union_intervals[mid][0] < c: 
                     # This interval starts before c. It might contain c. Store its index.
                     found_interval_idx = mid 
                     low = mid + 1
                 else:
                     # This interval starts at or after c. Check earlier intervals.
                     high = mid
            
            # Check the interval found at index `found_interval_idx` (the last one starting before c).
            if found_interval_idx != -1:
                prev_start, prev_end = union_intervals[found_interval_idx]
                # Check if c falls strictly between start and end of this interval.
                if prev_start < c < prev_end:
                    return True # c is inside (prev_start, prev_end), hence invalid cut.
            
            return False # c is not inside any open interval in the union, hence valid cut.

        # Function checks if two valid cuts can be made along a dimension
        def CanCut(N: int, intervals: List[List[int]]) -> bool:
            """
            Checks if two valid cuts c1, c2 (0 < c1 < c2 < N) can be made along the dimension 
            represented by intervals, such that each of the three resulting sections contains at least one rectangle.
            N is the grid dimension size. intervals are [start, end] for each rectangle in this dimension.
            """
            m = len(intervals)
            if m == 0: return False # Need rectangles to form sections

            # Compute the union of open intervals to identify regions where cuts are invalid.
            union_intervals = compute_union_open_intervals(intervals)

            # Collect potential cut locations: endpoints of intervals that are strictly between 0 and N.
            endpoints = set()
            for start, end in intervals:
                # A cut must be an integer at coordinate c where 0 < c < N.
                if 0 < start < N:
                    endpoints.add(start)
                if 0 < end < N:
                    endpoints.add(end)
            
            # Filter endpoints to keep only those that correspond to valid cut locations.
            ValidEndpoints = []
            for c in endpoints:
                 if not check_invalid(c, union_intervals):
                     ValidEndpoints.append(c)
            ValidEndpoints.sort() # Sort the valid endpoint coordinates.

            # If there are no valid cut locations among endpoints, we assume it's impossible based on this strategy.
            if not ValidEndpoints: 
                 return False

            # Calculate the minimum end coordinate and maximum start coordinate among all intervals.
            # These help determine if potential bottom/top sections can be non-empty.
            min_end = N + 1 # Initialize with a value larger than any possible coordinate
            max_start = -1 # Initialize with a value smaller than any possible coordinate
            for start, end in intervals:
                min_end = min(min_end, end)
                max_start = max(max_start, start)

            # C1: Set of valid endpoints 'c' such that there exists a rectangle ending at or before 'c'.
            # This implies that a cut at 'c' would result in a non-empty bottom/left section.
            # Equivalent condition: c >= min_end
            C1 = [c for c in ValidEndpoints if c >= min_end]
            
            # C2: Set of valid endpoints 'c' such that there exists a rectangle starting at or after 'c'.
            # This implies that a cut at 'c' would result in a non-empty top/right section.
            # Equivalent condition: c <= max_start
            C2 = [c for c in ValidEndpoints if c <= max_start]

            # If either C1 or C2 is empty, we cannot find cuts that satisfy boundary section non-empty conditions.
            if not C1 or not C2:
                 return False

            # Iterate through each rectangle 'k' to check if it can lie entirely within the middle section
            # defined by some pair of valid cuts (y1, y2) derived from C1 and C2.
            for k in range(m):
                start_k, end_k = intervals[k]
                
                # Find the largest valid cut y1 in C1 such that y1 <= start_k.
                # `bisect_right` finds insertion point for start_k. The element before it is the largest <= start_k.
                idx1 = bisect.bisect_right(C1, start_k)
                if idx1 == 0: continue # No element in C1 is <= start_k
                y1 = C1[idx1 - 1] 

                # Find the smallest valid cut y2 in C2 such that y2 >= end_k.
                # `bisect_left` finds insertion point for end_k. This index points to the smallest element >= end_k.
                idx2 = bisect.bisect_left(C2, end_k)
                if idx2 == len(C2): continue # No element in C2 is >= end_k
                y2 = C2[idx2] 
                
                # Ensure y1 < y2 for two distinct cuts. This condition must hold if start_k < end_k.
                # Since rectangles have positive width/height (start < end), this check is implicitly satisfied
                # if both y1 and y2 are found. However, explicitly checking `y1 < y2` adds robustness.
                if y1 < y2:
                    # Found a valid pair of cuts (y1, y2) such that:
                    # - y1 and y2 are valid cut locations.
                    # - Rectangle k is in the middle section [y1, y2).
                    # - Existence of y1 in C1 guarantees bottom/left section is non-empty.
                    # - Existence of y2 in C2 guarantees top/right section is non-empty.
                    # All conditions satisfied.
                    return True
            
            # If the loop completes without finding such a rectangle k and cuts y1, y2
            return False

        # Main logic: Check if cuts are possible horizontally, then vertically.
        # Prepare intervals for horizontal cuts (using y-coordinates)
        horizontal_intervals = [[r[1], r[3]] for r in rectangles] # [sy, ey]
        if CanCut(n, horizontal_intervals):
            return True # Found valid horizontal cuts

        # Prepare intervals for vertical cuts (using x-coordinates)
        vertical_intervals = [[r[0], r[2]] for r in rectangles] # [sx, ex]
        if CanCut(n, vertical_intervals):
            return True # Found valid vertical cuts
            
        # If neither horizontal nor vertical cuts work
        return False