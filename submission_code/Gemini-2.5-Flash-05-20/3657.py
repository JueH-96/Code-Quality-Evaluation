import bisect
from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        
        def _check(is_horizontal: bool) -> bool:
            events = []
            
            # Prepare data for section count checks (rectangles' start/end coordinates)
            # These are sorted to allow O(logN) lookup using bisect
            section_end_coords = []
            section_start_coords = []

            if is_horizontal:
                for rect in rectangles:
                    events.append((rect[1], +1)) # (start_y, type=+1)
                    events.append((rect[3], -1)) # (end_y, type=-1)
                    section_start_coords.append(rect[1])
                    section_end_coords.append(rect[3])
            else: # Vertical cuts
                for rect in rectangles:
                    events.append((rect[0], +1)) # (start_x, type=+1)
                    events.append((rect[2], -1)) # (end_x, type=-1)
                    section_start_coords.append(rect[0])
                    section_end_coords.append(rect[2])
            
            section_start_coords.sort()
            section_end_coords.sort()

            # Sort events: primary key is coordinate, secondary key is type (-1 before +1 for same coord)
            # Python's default tuple sort orders numbers correctly, so (-1) comes before (+1)
            events.sort()

            clear_cut_coords = []
            active_rect_count = 0
            event_idx = 0

            # Sweep-line to find all "clear lines"
            # A coordinate 'c' is a clear line if no rectangle strictly crosses 'c'.
            while event_idx < len(events):
                current_coord = events[event_idx][0]

                # Calculate temporary active count if all END events at current_coord were processed.
                # This count reflects the state *at* `current_coord` after items ending there leave.
                temp_active_for_check = active_rect_count
                temp_idx = event_idx
                while temp_idx < len(events) and events[temp_idx][0] == current_coord and events[temp_idx][1] == -1:
                    temp_active_for_check -= 1
                    temp_idx += 1
                
                # If `temp_active_for_check` is 0, it means no rectangles are active at this exact point (current_coord).
                # So, `current_coord` is a clear line where a cut can be made.
                if temp_active_for_check == 0:
                    clear_cut_coords.append(current_coord)

                # Now, process all events (both end and start) at current_coord to update the
                # `active_rect_count` for the interval immediately after `current_coord`.
                while event_idx < len(events) and events[event_idx][0] == current_coord:
                    active_rect_count += events[event_idx][1]
                    event_idx += 1
            
            # Filter clear_cut_coords: cuts must be strictly between 0 and n
            valid_clear_cuts = [c for c in clear_cut_coords if 0 < c < n]

            # We need at least two distinct valid cut coordinates
            if len(valid_clear_cuts) < 2:
                return False
            
            # Precompute counts for how many rectangles fall below a cut (Section 1)
            # and how many fall above a cut (Section 3).
            # `below_c_counts[k]` = count of rectangles 'r' where r.end_val <= valid_clear_cuts[k]
            # `above_c_counts[k]` = count of rectangles 'r' where r.start_val >= valid_clear_cuts[k]
            below_c_counts = [0] * len(valid_clear_cuts)
            above_c_counts = [0] * len(valid_clear_cuts)

            for k, c in enumerate(valid_clear_cuts):
                below_c_counts[k] = bisect.bisect_right(section_end_coords, c)
                above_c_counts[k] = len(rectangles) - bisect.bisect_left(section_start_coords, c)
            
            # Iterate through each rectangle, considering it as the required rectangle for the middle section (Section 2)
            for rect_b in rectangles:
                # Get the relevant start/end coordinates for the current orientation (horizontal/vertical)
                start_b = rect_b[1] if is_horizontal else rect_b[0]
                end_b = rect_b[3] if is_horizontal else rect_b[2]

                # Find candidate c1: largest clear_cut_coord <= start_b
                # bisect_right returns insertion point 'i' such that all `valid_clear_cuts[0...i-1]` are <= `start_b`
                c1_idx = bisect.bisect_right(valid_clear_cuts, start_b) - 1
                
                # Find candidate c2: smallest clear_cut_coord >= end_b
                # bisect_left returns insertion point 'i' such that all `valid_clear_cuts[i...]` are >= `end_b`
                c2_idx = bisect.bisect_left(valid_clear_cuts, end_b)
                
                # Check if valid indices are found and c1 < c2
                if c1_idx >= 0 and c2_idx < len(valid_clear_cuts) and c1_idx < c2_idx:
                    # Check if Section 1 (rects with end_val <= c1) has at least one rectangle
                    if below_c_counts[c1_idx] > 0:
                        # Check if Section 3 (rects with start_val >= c2) has at least one rectangle
                        if above_c_counts[c2_idx] > 0:
                            # Section 2 is guaranteed to have at least one rectangle (rect_b itself)
                            # because we selected c1 and c2 such that c1 <= start_b and end_b <= c2,
                            # and c1, c2 are clear lines, meaning rect_b belongs to the middle section.
                            return True # Found a valid set of cuts

            return False

        # Try horizontal cuts, if not possible, try vertical cuts
        return _check(True) or _check(False)