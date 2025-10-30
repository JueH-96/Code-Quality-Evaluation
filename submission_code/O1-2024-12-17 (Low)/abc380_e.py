def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse N, Q
    N, Q = map(int, input_data[:2])

    # We will maintain a list of "intervals" where each interval is (start, end, color).
    # All cells in [start, end] (inclusive) share the same color.
    # We will also maintain a parallel list "starts" that stores the "start" values
    # in strictly increasing order, so that we can binary-search to find which interval
    # a given cell x belongs to.

    # Initially, cell i has color i, so we have N intervals:
    # (1, 1, 1), (2, 2, 2), ..., (N, N, N)
    #
    # This means len(intervals) = N initially, which is large but still within 5*10^5.
    # We keep a color_count array of size N+1, where color_count[c] = number of cells
    # currently painted color c.

    # Then for a query of type "1 x c":
    # 1) Find the interval that contains x (via binary search in "starts").
    # 2) If that interval's color is already c, do nothing.
    #    Otherwise, let old_color be that interval's color. Decrease color_count[old_color]
    #    by the size of that interval, and increase color_count[c] by the same size.
    #    Recolor that entire interval to c.
    # 3) This may allow merging with the previous or next interval if they also have color c
    #    and are exactly adjacent. We implement a function merge_intervals(idx) to perform
    #    these merges in a loop until no further merges are possible.
    #
    # For a query "2 c", simply output color_count[c].

    sys.setrecursionlimit(10**7)

    # Read queries into a list for faster access
    ptr = 2
    queries = []
    for _ in range(Q):
        t = int(input_data[ptr]); ptr += 1
        if t == 1:
            x = int(input_data[ptr]); ptr += 1
            c = int(input_data[ptr]); ptr += 1
            queries.append((1, x, c))
        else:
            c = int(input_data[ptr]); ptr += 1
            queries.append((2, c))

    # Build the initial intervals
    intervals = [(i, i, i) for i in range(1, N+1)]
    starts = [i for i in range(1, N+1)]

    # Build color_count
    # Initially, each color i appears exactly once.
    color_count = [0] * (N+1)
    for i in range(1, N+1):
        color_count[i] = 1

    # Helper: find which interval index x belongs to
    def find_interval(x):
        # idx = largest i such that starts[i] <= x
        # We can do idx = bisect_right(starts, x) - 1
        idx = bisect.bisect_right(starts, x) - 1
        return idx

    # Helper: merge the interval at position idx with its neighbors if they share the same color
    def merge_intervals(idx):
        # Merge left
        while idx > 0:
            s0, e0, c0 = intervals[idx-1]
            s1, e1, c1 = intervals[idx]
            if c0 == c1 and e0 + 1 == s1:
                # Merge them
                new_interval = (s0, e1, c0)
                # Replace intervals[idx-1] with the merged
                intervals[idx-1] = new_interval
                starts[idx-1] = s0
                # Remove intervals[idx]
                intervals.pop(idx)
                starts.pop(idx)
                # Move idx back
                idx -= 1
            else:
                break

        # Merge right
        while idx < len(intervals)-1:
            s0, e0, c0 = intervals[idx]
            s1, e1, c1 = intervals[idx+1]
            if c0 == c1 and e0 + 1 == s1:
                # Merge them
                new_interval = (s0, e1, c0)
                intervals[idx] = new_interval
                starts[idx] = s0
                intervals.pop(idx+1)
                starts.pop(idx+1)
            else:
                break

    import sys
    out = []
    # Process queries
    for q in queries:
        if q[0] == 1:
            # 1 x c
            _, x, c = q
            idx = find_interval(x)
            s, e, old_color = intervals[idx]
            if old_color != c:
                length = e - s + 1
                color_count[old_color] -= length
                color_count[c] += length
                # Recolor
                intervals[idx] = (s, e, c)
                # Merge with neighbors if possible
                merge_intervals(idx)

        else:
            # 2 c
            _, c = q
            out.append(str(color_count[c]))

    print("
".join(out))

# Do not forget to call main()
main()