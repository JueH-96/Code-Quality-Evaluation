def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    M = int(input_data[1])
    Sx = int(input_data[2])
    Sy = int(input_data[3])
    ptr = 4

    houses = []
    for i in range(N):
        x = int(input_data[ptr]); y = int(input_data[ptr+1])
        ptr += 2
        houses.append((x, y))

    moves = []
    for i in range(M):
        d = input_data[ptr]
        c = int(input_data[ptr+1])
        ptr += 2
        moves.append((d, c))

    # ---------------------------------------------------------
    # Build the visited-path as intervals:
    # We'll keep two dicts:
    #   vert[x]   = list of (y1, y2) intervals (in real sense, inclusive)
    #   horiz[y]  = list of (x1, x2) intervals (in real sense, inclusive)
    #
    # Then we'll merge intervals for each x or y.
    # Finally, each house (xh, yh) is visited if
    #   xh in vert and yh is in one of vert[xh]'s intervals
    #   OR
    #   yh in horiz and xh is in one of horiz[yh]'s intervals
    # We'll do a binary search to check membership in intervals.

    from collections import defaultdict

    vert = defaultdict(list)   # x -> list of (y1, y2)
    horiz = defaultdict(list)  # y -> list of (x1, x2)

    # Build the path (sequence of positions)
    curx, cury = Sx, Sy
    for (d, c) in moves:
        oldx, oldy = curx, cury
        if d == 'U':
            cury += c
            x0 = oldx
            y1, y2 = (oldy, cury) if oldy <= cury else (cury, oldy)
            vert[x0].append((y1, y2))
        elif d == 'D':
            cury -= c
            x0 = oldx
            y1, y2 = (cury, oldy) if cury <= oldy else (oldy, cury)
            vert[x0].append((y1, y2))
        elif d == 'R':
            curx += c
            y0 = oldy
            x1, x2 = (oldx, curx) if oldx <= curx else (curx, oldx)
            horiz[y0].append((x1, x2))
        else:  # d == 'L'
            curx -= c
            y0 = oldy
            x1, x2 = (curx, oldx) if curx <= oldx else (oldx, curx)
            horiz[y0].append((x1, x2))

    # At this point, (curx, cury) is the final position.
    final_x, final_y = curx, cury

    # ---------------------------------------------------------
    # Function to merge intervals in-place
    def merge_intervals(intervals):
        # intervals is a list of (start, end)
        # returns a new sorted list of non-overlapping intervals
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                prev_start, prev_end = merged[-1]
                cur_start, cur_end = interval
                if cur_start <= prev_end:
                    # overlap => merge
                    merged[-1] = (prev_start, max(prev_end, cur_end))
                else:
                    # no overlap => push new
                    merged.append(interval)
        return merged

    # Merge intervals in vert
    for x0 in list(vert.keys()):
        vert[x0] = merge_intervals(vert[x0])

    # Merge intervals in horiz
    for y0 in list(horiz.keys()):
        horiz[y0] = merge_intervals(horiz[y0])

    # ---------------------------------------------------------
    # We now define a function to check if val is in
    # the union of intervals = intervals_list (which is sorted, non-overlapping).
    # We'll do a binary search to find the position of (val, val) insertion
    import bisect

    def in_union(intervals_list, val):
        # intervals_list is sorted by start
        # We want to quickly check if val is in any (start, end) in intervals_list.
        # We'll do a bisect to find the rightmost interval whose start <= val
        # and see if that interval's end >= val
        #   i = bisect_right by the interval start
        # Actually we can do:
        #   i = bisect.bisect_right(intervals_list, (val, float('inf'))) - 1
        #   if i >= 0, check intervals_list[i] if it covers val
        # intervals_list[i] = (start, end)
        # we check if start <= val <= end.
        i = bisect.bisect_right(intervals_list, (val, float('inf'))) - 1
        if i < 0:
            return False
        start_i, end_i = intervals_list[i]
        return (start_i <= val <= end_i)

    # ---------------------------------------------------------
    # Finally, we check for each house whether it's visited.
    visited_count = 0
    for (xh, yh) in houses:
        # Check if xh in vert, and if yh in union(vert[xh])
        # or if yh in horiz, and if xh in union(horiz[yh])
        # If yes, visited_count += 1
        intervals_v = vert.get(xh, None)
        if intervals_v is not None:
            if in_union(intervals_v, yh):
                visited_count += 1
                continue
        intervals_h = horiz.get(yh, None)
        if intervals_h is not None:
            if in_union(intervals_h, xh):
                visited_count += 1
                continue

    # Output
    print(final_x, final_y, visited_count)

# Don't forget to call main().
if __name__ == "__main__":
    main()