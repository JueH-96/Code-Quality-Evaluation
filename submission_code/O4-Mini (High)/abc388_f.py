import sys
import threading

def main():
    import sys
    from bisect import bisect_left, bisect_right

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = int(next(it))
    B = int(next(it))
    bad = []
    for _ in range(M):
        l = int(next(it)); r = int(next(it))
        bad.append((l, r))
    # Build good intervals: segments of [l_i, r_i] that are not bad
    good = []
    prev = 1
    for (l, r) in bad:
        if prev <= l-1:
            good.append((prev, l-1))
        prev = r+1
    if prev <= N:
        good.append((prev, N))
    # Prepare reachable intervals as sorted non-overlapping list
    reachable_starts = [1]
    reachable_ends   = [1]
    # DP parameters
    W = B - A + 1
    # scanning limit per block: B * W
    max_scan = B * W

    # Function to merge a new interval [s,e] into reachable_intervals
    def add_interval(s, e):
        # merge into reachable_starts, reachable_ends
        starts = reachable_starts
        ends = reachable_ends
        # find position to insert by start
        pos = bisect_left(starts, s)
        new_s = s
        new_e = e
        # merge with left
        if pos > 0 and ends[pos-1] >= s-1:
            pos -= 1
            new_s = starts[pos]
            if ends[pos] > new_e:
                new_e = ends[pos]
            # remove the old interval
            del starts[pos]
            del ends[pos]
        # merge with right
        while pos < len(starts) and starts[pos] <= new_e + 1:
            # overlapping or adjacent
            if ends[pos] > new_e:
                new_e = ends[pos]
            del starts[pos]
            del ends[pos]
        # insert the merged new interval
        starts.insert(pos, new_s)
        ends.insert(pos, new_e)

    # Process each good interval
    for (l, r) in good:
        # local reachable in this block
        local_starts = []
        local_ends   = []
        # scanning bounds
        # dp[1] is true by initial condition, so we start scanning from x>=2
        x_start = l if l > 1 else 2
        if x_start > r:
            # this block only contains 1 which is already in global; skip
            continue
        scan_end = r if r <= l + max_scan else (l + max_scan)
        run = 0
        sat = False
        sat_pos = -1
        # cache for globals
        g_starts = reachable_starts
        g_ends   = reachable_ends
        # local lists for binary search
        # We'll use bisect on local_starts, local_ends
        for x in range(x_start, scan_end + 1):
            # check window [x-B, x-A]
            lo = x - B
            hi = x - A
            reachable = False
            # check global intervals
            # find rightmost interval with start <= hi
            idx = bisect_right(g_starts, hi) - 1
            if idx >= 0 and g_ends[idx] >= lo:
                reachable = True
            else:
                # check local intervals
                if local_starts:
                    idx2 = bisect_right(local_starts, hi) - 1
                    if idx2 >= 0 and local_ends[idx2] >= lo:
                        reachable = True
            if reachable:
                run += 1
                # add to local intervals
                if local_starts and local_ends[-1] == x - 1:
                    local_ends[-1] += 1
                else:
                    local_starts.append(x)
                    local_ends.append(x)
                if run >= W:
                    sat = True
                    sat_pos = x
                    break
            else:
                run = 0
        # if no reachable in this block, skip
        if not local_starts:
            continue
        # if saturated, add tail [sat_pos+1, r]
        if sat:
            tail = sat_pos + 1
            if tail <= r:
                # merge tail
                if local_ends[-1] == sat_pos:
                    local_ends[-1] = r
                else:
                    local_starts.append(tail)
                    local_ends.append(r)
        # else if we scanned full block (scan_end == r), local_intervals complete
        # else scan_end < r and no sat: we do not assume tail reachable

        # merge local intervals into global reachable
        # local_starts/local_ends sorted, non-overlapping
        for s, e in zip(local_starts, local_ends):
            add_interval(s, e)

    # After all, check if N is reachable
    idx = bisect_right(reachable_starts, N) - 1
    if idx >= 0 and reachable_ends[idx] >= N:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()