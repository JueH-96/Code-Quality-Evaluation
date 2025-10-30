import sys
# sortedcontainers is often available in competitive programming environments
# If not available, a custom balanced BST or segment tree implementation would be needed.
from sortedcontainers import SortedList

def solve():
    H, W, Q = map(int, sys.stdin.readline().split())

    # wall_rows[c] is a SortedList of row indices 'r' where (r, c) is a wall.
    # wall_cols[r] is a SortedList of col indices 'c' where (r, c) is a wall.
    wall_rows = [SortedList(range(H)) for _ in range(W)]
    wall_cols = [SortedList(range(W)) for _ in range(H)]

    # destroyed_row_segments[c] stores (start_r, end_r) tuples for destroyed cells in column 'c'.
    # destroyed_col_segments[r] stores (start_c, end_c) tuples for destroyed cells in row 'r'.
    # These segments represent contiguous blocks of destroyed cells.
    destroyed_row_segments = [SortedList() for _ in range(W)]
    destroyed_col_segments = [SortedList() for _ in range(H)]

    current_walls_count = H * W

    # Helper function to check if a range (query_start, query_end) is completely covered by
    # one segment in the given SortedList of segments.
    # An empty range (query_start > query_end) is considered fully destroyed.
    def is_range_fully_destroyed(segments_list, query_start, query_end):
        if query_start > query_end: 
            return True
        
        # Find the segment that potentially contains query_start.
        # We search for the first segment (s,e) such that s >= query_start + 1.
        # The segment just before this index (if it exists) is the candidate to cover query_start.
        idx = segments_list.bisect_left((query_start + 1, -1)) 
        
        if idx > 0:
            s, e = segments_list[idx - 1]
            # Check if this candidate segment actually covers the whole query range.
            if s <= query_start and e >= query_end:
                return True
        return False

    # This function attempts to destroy a wall at (r,c) and updates counts and segments.
    # Returns True if a wall was successfully destroyed, False otherwise (e.g., no wall there or out of bounds).
    def destroy_and_update_segments(r, c):
        nonlocal current_walls_count
        if r < 0 or r >= H or c < 0 or c >= W:
            return False

        # Check if it's currently a wall at (r,c) using wall_cols (which implies it's in wall_rows too).
        if c in wall_cols[r]: 
            # Remove (r,c) from wall tracking lists
            wall_cols[r].remove(c)
            wall_rows[c].remove(r)
            current_walls_count -= 1

            # Update destroyed row segments in column 'c'
            segments_list_r = destroyed_row_segments[c]
            start_merged_r = r
            end_merged_r = r
            segments_to_remove_r = []

            # Check for segment ending at r-1 (segment immediately above)
            # bisect_left((r, r)) gives the insertion point for (r,r), so (idx-1) checks before it.
            idx_prev_r = segments_list_r.bisect_left((r, r))
            if idx_prev_r > 0:
                s_prev, e_prev = segments_list_r[idx_prev_r - 1]
                if e_prev == r - 1: # Found adjacent segment above
                    start_merged_r = s_prev
                    segments_to_remove_r.append((s_prev, e_prev))
                # If s_prev <= r <= e_prev, it means (r,c) was already part of a segment.
                # This case should ideally not happen if 'c in wall_cols[r]' check passes,
                # but it's a good safety check in general segment merging.

            # Check for segment starting at r+1 (segment immediately below)
            # bisect_right((r, r)) gives insertion point for (r,r)+1 conceptually.
            idx_next_r = segments_list_r.bisect_right((r, r))
            if idx_next_r < len(segments_list_r):
                s_next, e_next = segments_list_r[idx_next_r]
                if s_next == r + 1: # Found adjacent segment below
                    end_merged_r = e_next
                    segments_to_remove_r.append((s_next, e_next))
            
            # Remove old segments and add the new merged segment
            for seg in segments_to_remove_r:
                segments_list_r.remove(seg)
            segments_list_r.add((start_merged_r, end_merged_r))

            # Update destroyed column segments in row 'r' (symmetric logic)
            segments_list_c = destroyed_col_segments[r]
            start_merged_c = c
            end_merged_c = c
            segments_to_remove_c = []

            idx_prev_c = segments_list_c.bisect_left((c, c))
            if idx_prev_c > 0:
                s_prev_c, e_prev_c = segments_list_c[idx_prev_c - 1]
                if e_prev_c == c - 1:
                    start_merged_c = s_prev_c
                    segments_to_remove_c.append((s_prev_c, e_prev_c))
            
            idx_next_c = segments_list_c.bisect_right((c, c))
            if idx_next_c < len(segments_list_c):
                s_next_c, e_next_c = segments_list_c[idx_next_c]
                if s_next_c == c + 1:
                    end_merged_c = e_next_c
                    segments_to_remove_c.append((s_next_c, e_next_c))
            
            for seg_c in segments_to_remove_c:
                segments_list_c.remove(seg_c)
            segments_list_c.add((start_merged_c, end_merged_c))
            
            return True # Wall was destroyed
        return False # No wall at (r,c) to destroy

    for _ in range(Q):
        R_q, C_q = map(int, sys.stdin.readline().split())
        r, c = R_q - 1, C_q - 1 # Convert to 0-indexed

        if destroy_and_update_segments(r, c):
            # Case 1: (r,c) was a wall and has been destroyed. Query process ends for this query.
            pass
        else:
            # Case 2: (r,c) was already destroyed. Find walls in 4 directions.
            walls_to_destroy_in_this_query = set() # Use a set to avoid adding duplicates if multiple paths lead to same wall

            # 1. Up: Find largest i < r such that (i,c) is a wall and (i+1..r-1, c) are destroyed.
            idx = wall_rows[c].bisect_left(r)
            if idx > 0: # There is at least one wall above 'r' in column 'c'
                candidate_r = wall_rows[c][idx - 1]
                # Check if the range between candidate_r and r (exclusive) is fully destroyed.
                if is_range_fully_destroyed(destroyed_row_segments[c], candidate_r + 1, r - 1):
                    walls_to_destroy_in_this_query.add((candidate_r, c))

            # 2. Down: Find smallest i > r such that (i,c) is a wall and (r+1..i-1, c) are destroyed.
            idx = wall_rows[c].bisect_left(r + 1)
            if idx < len(wall_rows[c]): # There is at least one wall below 'r' in column 'c'
                candidate_r = wall_rows[c][idx]
                if is_range_fully_destroyed(destroyed_row_segments[c], r + 1, candidate_r - 1):
                    walls_to_destroy_in_this_query.add((candidate_r, c))

            # 3. Left: Find largest j < c such that (r,j) is a wall and (r, j+1..c-1) are destroyed.
            idx = wall_cols[r].bisect_left(c)
            if idx > 0: # There is at least one wall to the left of 'c' in row 'r'
                candidate_c = wall_cols[r][idx - 1]
                if is_range_fully_destroyed(destroyed_col_segments[r], candidate_c + 1, c - 1):
                    walls_to_destroy_in_this_query.add((r, candidate_c))

            # 4. Right: Find smallest j > c such that (r,j) is a wall and (r, c+1..j-1) are destroyed.
            idx = wall_cols[r].bisect_left(c + 1)
            if idx < len(wall_cols[r]): # There is at least one wall to the right of 'c' in row 'r'
                candidate_c = wall_cols[r][idx]
                if is_range_fully_destroyed(destroyed_col_segments[r], c + 1, candidate_c - 1):
                    walls_to_destroy_in_this_query.add((r, candidate_c))
            
            # Destroy all identified walls for this query.
            for wr, wc in walls_to_destroy_in_this_query:
                destroy_and_update_segments(wr, wc)

    sys.stdout.write(str(current_walls_count) + '
')

# Call the solve function to run the program
solve()