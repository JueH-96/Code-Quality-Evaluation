import sys

def main():
    N = int(sys.stdin.readline())
    
    events = []
    # Using original 0-based index from input reading order as interval_idx
    for i in range(N):
        l, r = map(int, sys.stdin.readline().split())
        # Event type: 0 for start, 1 for end
        # An interval is [l, r] inclusive.
        # A start event occurs at 'l'.
        # An end event occurs at 'r'.
        events.append((l, 0, i)) 
        events.append((r, 1, i))

    # Sort events:
    # 1. By coordinate.
    # 2. If coordinates are equal, by type: 0 (start) before 1 (end).
    #    This ensures that at any coordinate C:
    #    - All intervals starting at C are processed. For each, it's added to active_set
    #      and its intersections with already-active intervals are counted.
    #    - Then, all intervals ending at C are processed (removed from active_set).
    #    This correctly handles cases like [1,5] and [5,10] intersecting at point 5.
    #    Interval [1,5] is still active when [5,10] starts.
    # 3. If coordinates and types are also same (e.g., multiple starts at the same coordinate):
    #    The third element of the tuple (interval_idx) will be used as a tie-breaker.
    #    The order among same-type events at the same coordinate does not affect the final count.
    events.sort()

    active_set = set() # Stores interval_idx of currently active intervals
    intersection_count = 0

    for coord, type, interval_idx in events:
        if type == 0:  # This is a start event for interval 'interval_idx'
            # This interval [l_interval_idx, r_interval_idx] starts at 'coord'.
            # It intersects with every interval 'k' that is currently in active_set.
            # Proof: For any k in active_set:
            #   1. l_k <= coord (k started at or before 'coord' and its start event was processed)
            #   2. r_k >= coord (k has not ended yet. If r_k == coord, its end event (type 1)
            #                  is processed after this start event (type 0) due to sorting)
            # Conditions 1 and 2 mean 'coord' is within [l_k, r_k].
            # Since 'coord' is the start point of interval_idx, 'coord' is also in interval_idx.
            # Thus, interval_idx and k share the point 'coord', so they intersect.
            intersection_count += len(active_set)
            active_set.add(interval_idx)
        else:  # This is an end event for interval 'interval_idx'
            # Interval 'interval_idx' (which ends at 'coord')
            # is no longer active *after* this coordinate point.
            # It must be in active_set because its start event must have been processed earlier.
            # (Constraint l_i < r_i means an interval's start coord is strictly less than its end coord.
            # Thus, start event is always processed before end event for the same interval).
            active_set.remove(interval_idx) # Raises KeyError if not found, but logic implies it's always present.

    print(intersection_count)

if __name__ == '__main__':
    main()