import sys
from sortedcontainers import SortedList

# Function to check if two intervals [s1, e1) and [s2, e2) overlap
def check_overlap(interval1_start, interval1_end_exclusive, interval2_start, interval2_end_exclusive):
    return max(interval1_start, interval2_start) < min(interval1_end_exclusive, interval2_end_exclusive)

# Function to check if a query interval overlaps with any interval in a SortedList
# SortedList stores intervals as (start, end_exclusive) tuples, sorted by start.
# This implementation is O(log N + k) where k is the number of intervals that start before query_end_exclusive and need checking.
# In the worst case, k can be O(N). However, for sparse distributions or limited overlap, it might be faster.
# A proper interval tree query is O(log W).
def has_overlap_efficient(intervals_sl, query_start, query_end_exclusive):
    # Find the index of the first interval whose start is >= query_start
    # This is the potential starting point for intervals overlapping from the left or fully within the query.
    idx_check_curr = intervals_sl.bisect_left((query_start, -1))

    # Check intervals starting from idx_check_curr onwards
    # These intervals [s, e) have s >= query_start. They overlap if s < query_end_exclusive.
    for idx in range(idx_check_curr, len(intervals_sl)):
        s, e = intervals_sl[idx]
        # If the current interval starts after the query ends, no subsequent interval (which start even later) will overlap either.
        if s >= query_end_exclusive:
            break
        # The interval starts before query_end_exclusive. Overlap exists if it ends after query_start.
        if e > query_start:
            return True # Overlap found

    # Check the interval immediately before idx_check_curr (if it exists)
    # This interval [s, e) has s < query_start. It overlaps if its end e > query_start.
    if idx_check_curr > 0:
        s, e = intervals_sl[idx_check_curr - 1]
        if e > query_start:
            return True # Overlap found

    return False # No overlap found

def solve():
    # Read input
    H, W, N = map(int, sys.stdin.readline().split())
    bars_info = []
    # Store initial bar info indexed by original bar number (0 to N-1)
    initial_R = [0] * N
    initial_C = [0] * N
    initial_L = [0] * N
    for i in range(N):
        R, C, L = map(int, sys.stdin.readline().split())
        # Store 0-indexed C (columns are 1-based in problem, use 0-based internally)
        bars_info.append((R, C - 1, L, i)) # Store (R, C_0based, L, original_index)
        initial_R[i] = R
        initial_C[i] = C - 1 # Store 0-based column start
        initial_L[i] = L

    # R_current stores the current row of each bar (0-indexed bar index -> 1-indexed row)
    R_current = list(initial_R)

    # Map: row (1-indexed) -> SortedList of intervals [col_start_0based, col_end_exclusive_0based)
    row_intervals_sl = {}

    # Initial state: Populate row_intervals_sl from initial positions
    for i in range(N):
        r = R_current[i]
        c = initial_C[i] # 0-based start column
        l = initial_L[i]
        if r not in row_intervals_sl:
            row_intervals_sl[r] = SortedList()
        row_intervals_sl[r].add((c, c + l)) # Store [c, c+l)

    # Simulation loop
    stable = False
    while not stable:
        stable = True # Assume stable for this round until a bar moves

        # Iterate through each bar in order 0 to N-1 (corresponding to bar number 1 to N)
        for i in range(N):
            current_r = R_current[i] # Row of bar i at the start of its turn in this round (1-indexed)
            c = initial_C[i] # 0-based start column
            l = initial_L[i]
            target_r = current_r + 1 # The row bar i wants to move to (1-indexed)

            # Check if bar is already on the bottom row (H-th row)
            if current_r == H:
                 continue # Bar is on the bottom row, cannot move down

            # Check if the target row `target_r` is occupied by any bar k and overlaps horizontally with bar i.
            # The state (R_current and row_intervals_sl) reflects moves by bars 0..i-1 in this round.
            is_blocked = False

            # Check if the target row `target_r` currently has any bars
            if target_r in row_intervals_sl:
                # Check for horizontal overlap with any bar located at target_r
                # Bar i occupies columns [c, c+l-1]. Query interval is [c, c+l) in 0-based.
                if has_overlap_efficient(row_intervals_sl[target_r], c, c + l):
                    is_blocked = True # Found a blocking bar

            if not is_blocked:
                # Bar i can move down by one cell
                stable = False # A bar moved, the system is not yet stable

                # Update R_current for bar i
                R_current[i] = target_r

                # Update row_intervals_sl: remove bar i's interval from its old row and add to its new row

                # The interval representing bar i
                bar_interval = (c, c + l)

                # Remove from old row (current_r)
                # Check if current_r is still in row_intervals_sl (it should be if R_current[i] is valid)
                if current_r in row_intervals_sl:
                    # Remove the specific interval for bar i from the SortedList for `current_r`
                    # Note: `remove` on SortedList removes the first occurrence if duplicates exist.
                    # Since bars start at non-overlapping positions, this should be fine.
                    row_intervals_sl[current_r].remove(bar_interval)
                    # If the SortedList for current_r becomes empty, remove the row key from the map
                    if not row_intervals_sl[current_r]:
                        del row_intervals_sl[current_r]
                # else: Should not happen in a consistent state.

                # Add to new row (target_r)
                if target_r not in row_intervals_sl:
                    # Create a new SortedList for the target row if it doesn't exist
                    row_intervals_sl[target_r] = SortedList()
                # Add the interval to the new row's list
                row_intervals_sl[target_r].add(bar_interval)

        # End of the round (finished processing bars 0 to N-1)
        # The loop continues if any bar moved in this round (stable is False)

    # Simulation finished when stable is True (no bar moved in the last round)
    # Print the final rows (1-indexed) of the bars
    for r in R_current:
        print(r)

# Execute the solve function
solve()