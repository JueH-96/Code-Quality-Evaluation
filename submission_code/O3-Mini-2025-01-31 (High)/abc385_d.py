def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    # Read first line: number of houses, number of moves, and starting position.
    N = int(next(it))
    M = int(next(it))
    Sx = int(next(it))
    Sy = int(next(it))
    
    # Read house coordinates.
    houses = [ (int(next(it)), int(next(it))) for _ in range(N) ]
    
    # We will record Santa's moves as line segments.
    # For vertical moves (U, D), we record segments along a fixed x.
    # For horizontal moves (L, R), we record segments along a fixed y.
    vertical_segments_by_x = {}
    horizontal_segments_by_y = {}
    
    cx, cy = Sx, Sy
    for _ in range(M):
        d = next(it)
        c = int(next(it))
        if d == 'U':
            ny = cy + c
            lo = cy if cy < ny else ny
            hi = ny if ny > cy else cy
            vertical_segments_by_x.setdefault(cx, []).append((lo, hi))
            cy = ny
        elif d == 'D':
            ny = cy - c
            lo = ny if ny < cy else cy
            hi = cy if cy > ny else ny
            vertical_segments_by_x.setdefault(cx, []).append((lo, hi))
            cy = ny
        elif d == 'L':
            nx = cx - c
            lo = nx if nx < cx else cx
            hi = cx if cx > nx else nx
            horizontal_segments_by_y.setdefault(cy, []).append((lo, hi))
            cx = nx
        elif d == 'R':
            nx = cx + c
            lo = cx if cx < nx else nx
            hi = nx if nx > cx else cx
            horizontal_segments_by_y.setdefault(cy, []).append((lo, hi))
            cx = nx
        else:
            # Should not occur.
            pass

    # The final position of Santa.
    final_x, final_y = cx, cy

    # For efficient queries later, merge overlapping segments along each line.
    # Merging vertical segments for each x coordinate.
    for x in vertical_segments_by_x:
        segs = vertical_segments_by_x[x]
        segs.sort(key=lambda seg: seg[0])
        merged = []
        for seg in segs:
            if not merged:
                merged.append(seg)
            else:
                last_lo, last_hi = merged[-1]
                if seg[0] <= last_hi:
                    merged[-1] = (last_lo, max(last_hi, seg[1]))
                else:
                    merged.append(seg)
        vertical_segments_by_x[x] = merged

    # Merging horizontal segments for each y coordinate.
    for y in horizontal_segments_by_y:
        segs = horizontal_segments_by_y[y]
        segs.sort(key=lambda seg: seg[0])
        merged = []
        for seg in segs:
            if not merged:
                merged.append(seg)
            else:
                last_lo, last_hi = merged[-1]
                if seg[0] <= last_hi:
                    merged[-1] = (last_lo, max(last_hi, seg[1]))
                else:
                    merged.append(seg)
        horizontal_segments_by_y[y] = merged

    # To do fast binary search for a point in an interval list, precompute the list of starting endpoints.
    vert_lookup = {}
    for x, segs in vertical_segments_by_x.items():
        starts = [lo for lo, hi in segs]
        ends = [hi for lo, hi in segs]
        vert_lookup[x] = (starts, ends)
        
    horiz_lookup = {}
    for y, segs in horizontal_segments_by_y.items():
        starts = [lo for lo, hi in segs]
        ends = [hi for lo, hi in segs]
        horiz_lookup[y] = (starts, ends)
    
    # Check each house to see if it was visited.
    visited_count = 0
    for x, y in houses:
        hit = False
        # If the house lies on some vertical segment (i.e. same x coordinate)
        if x in vert_lookup:
            starts, ends = vert_lookup[x]
            pos = bisect.bisect_right(starts, y)
            if pos and y <= ends[pos-1]:
                hit = True
        # If not found vertically, check horizontal segments (i.e. same y coordinate).
        if not hit and y in horiz_lookup:
            starts, ends = horiz_lookup[y]
            pos = bisect.bisect_right(starts, x)
            if pos and x <= ends[pos-1]:
                hit = True
        if hit:
            visited_count += 1

    # Print final position and number of houses visited.
    sys.stdout.write(f"{final_x} {final_y} {visited_count}")

if __name__ == '__main__':
    main()