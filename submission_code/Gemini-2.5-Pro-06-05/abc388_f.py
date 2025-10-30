import sys
from collections import deque

def merge_intervals(intervals):
    """
    Merges a list of possibly overlapping intervals into a sorted list of disjoint intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals based on their start point
    intervals.sort()
    
    merged = []
    if not intervals:
        return merged
        
    current_start, current_end = intervals[0]
    
    for next_start, next_end in intervals[1:]:
        if next_start <= current_end + 1:
            # The next interval overlaps or is adjacent to the current one, merge them
            current_end = max(current_end, next_end)
        else:
            # The next interval starts a new disjoint block
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
            
    merged.append((current_start, current_end))
    return merged

def solve():
    """
    Main function to solve the problem.
    """
    N, M, A, B = map(int, sys.stdin.readline().split())
    
    bad_ranges = []
    for _ in range(M):
        bad_ranges.append(tuple(map(int, sys.stdin.readline().split())))
    
    # Define the good intervals based on the bad ones
    good_intervals = []
    last_r = 0
    for l_bad, r_bad in bad_ranges:
        # The good interval is the gap between the last bad one and the current one
        g_start = last_r + 1
        g_end = l_bad - 1
        if g_start <= g_end:
            good_intervals.append((g_start, g_end))
        last_r = r_bad
        
    # Add the final good interval from the last bad one to N
    if last_r < N:
        good_intervals.append((last_r + 1, N))

    # `reachable` stores the set of all reachable squares as a list of disjoint intervals
    reachable = [(1, 1)]

    # Process each good interval from left to right
    for g_start, g_end in good_intervals:
        # Phase 1: Propagate. Find what's reachable in this zone from previous zones.
        propagated = []
        for s, e in reachable:
            # Projected range from a previous reachable interval
            p_start, p_end = s + A, e + B
            
            # Intersection with the current good interval
            i_start = max(g_start, p_start)
            i_end = min(g_end, p_end)
            
            if i_start <= i_end:
                propagated.append((i_start, i_end))
        
        # Merge propagated intervals to get the initial foothold in this zone.
        initial_in_zone = merge_intervals(propagated)

        if not initial_in_zone:
            # If we cannot jump into this zone, we might be stuck.
            # But we might have already been in an interval spanning this zone.
            # We check if any part of `reachable` is already in this good interval.
            is_already_in_zone = False
            for s, e in reachable:
                if max(g_start, s) <= min(g_end, e):
                    is_already_in_zone = True
                    break
            if not is_already_in_zone:
                # Can't reach this zone at all, so can't reach N unless we already have.
                can_reach_n = any(e == N for s, e in reachable)
                if not can_reach_n:
                    print("No")
                    return

        # Phase 2: Expand. Find all reachable squares within this zone.
        # This is a fixed-point iteration (or a BFS on intervals).
        q = deque(initial_in_zone)
        processed_in_zone = set(initial_in_zone)
        
        while q:
            s, e = q.popleft()
            
            p_start, p_end = s + A, e + B
            i_start = max(g_start, p_start)
            i_end = min(g_end, p_end)
            
            if i_start <= i_end:
                new_interval = (i_start, i_end)
                if new_interval not in processed_in_zone:
                    q.append(new_interval)
                    processed_in_zone.add(new_interval)

        # The new total set of reachable intervals is the union of the old set
        # and what we found by expanding in the current zone.
        reachable = merge_intervals(reachable + list(processed_in_zone))

    # After processing all good intervals, check if N is in the final reachable set.
    if reachable and reachable[-1][1] == N:
        print("Yes")
    else:
        print("No")

solve()