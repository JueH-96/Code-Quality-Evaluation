def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse N and Q.
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    
    # Initially, each cell i (from 1 to N) has its own color i.
    # Thus, we have N intervals: (l, r, col) for each cell.
    intervals = [(i, i, i) for i in range(1, N+1)]
    
    # Maintain a count for each color (colors range from 1 to N).
    counts = [0] * (N + 1)
    for i in range(1, N + 1):
        counts[i] = 1
    
    # A helper function to find the index of the interval that contains cell x.
    # Since intervals are stored as (l, r, col) sorted by l and are contiguous and non-overlapping,
    # we can binary search by the left endpoint.
    def find_interval(x):
        lo, hi = 0, len(intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            if intervals[mid][0] > x:
                hi = mid
            else:
                lo = mid + 1
        return lo - 1

    out_lines = []
    
    # Process the queries.
    for _ in range(Q):
        typ = next(it)
        if typ == b'1':  # Query type 1: "1 x c"
            x = int(next(it))
            new_color = int(next(it))
            idx = find_interval(x)
            l, r, old_color = intervals[idx]
            # If the region (the entire interval that contains x) is already the desired color, do nothing.
            if old_color == new_color:
                continue
            region_length = r - l + 1
            # Update counts.
            counts[old_color] -= region_length
            counts[new_color] += region_length
            # Repaint the entire interval.
            intervals[idx] = (l, r, new_color)
            # Merge with a left neighbor if it has the same color and is contiguous.
            while idx > 0:
                l_prev, r_prev, col_prev = intervals[idx - 1]
                if col_prev == new_color and r_prev + 1 == intervals[idx][0]:
                    # Merge the left interval with the current one.
                    new_l = l_prev
                    new_r = intervals[idx][1]
                    intervals[idx - 1] = (new_l, new_r, new_color)
                    del intervals[idx]
                    idx -= 1
                else:
                    break
            # Merge with a right neighbor if it has the same color and is contiguous.
            while idx < len(intervals) - 1:
                l_next, r_next, col_next = intervals[idx + 1]
                if col_next == new_color and intervals[idx][1] + 1 == l_next:
                    new_l = intervals[idx][0]
                    new_r = r_next
                    intervals[idx] = (new_l, new_r, new_color)
                    del intervals[idx + 1]
                else:
                    break
        else:  # Query type 2: "2 c"
            c = int(next(it))
            out_lines.append(str(counts[c]))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()