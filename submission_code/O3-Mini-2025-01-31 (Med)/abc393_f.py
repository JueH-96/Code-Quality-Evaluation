def main():
    import sys,sys
    from bisect import bisect_left, bisect_right
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
        q = int(next(it))
    except StopIteration:
        return
    # Read the sequence A; note that our A is 0-indexed but queries use 1-indexed positions.
    A = [int(next(it)) for _ in range(n)]
    
    # We simulate the standard patience algorithm.
    # dp will always hold the current tail values (in increasing order). For each "slot" (or position in dp)
    # we record the "update events". An event (time, value) means that at that time (i+1) the dp slot becomes value.
    dp = []  
    events = []  # events[i] for slot i (0-indexed) is a list of (time, new_val)
    for i, a in enumerate(A):
        pos = bisect_left(dp, a)
        if pos == len(dp):
            dp.append(a)
            events.append([(i+1, a)])
        else:
            dp[pos] = a
            events[pos].append((i+1, a))
    m = len(events)  # overall number of slots (i.e. maximum length of an increasing subsequence on the full sequence)
    
    # For each slot, extract separate lists: one for update times and one for corresponding dp values.
    slot_times = [ [ ev[0] for ev in events[i] ] for i in range(m) ]
    slot_vals  = [ [ ev[1] for ev in events[i] ] for i in range(m) ]
    # Also for each slot its very first update time.
    first_times = [ events[i][0][0] for i in range(m) ]
    
    # Helper: given a slot index (0-indexed) and a prefix time R, 
    # return the dp value in that slot for the prefix.
    INF = 10**12
    def get_slot_value(slot, R):
        # We want the last event in slot_times[slot] with time <= R.
        arr = slot_times[slot]
        idx = bisect_right(arr, R)
        if idx == 0:
            return INF
        return slot_vals[slot][idx-1]
    
    # Now answer each query.
    # For query (R, X): let Lmax be the number of slots j for which first_times[j] ≤ R.
    # Then in the prefix dp snapshot (dp_R) the "profile" has length Lmax.
    # Because dp_R is increasing we simply need the largest j (0-indexed) for which dp_R[j] ≤ X.
    # That is, answer = (number of slots j in [0,Lmax) with get_slot_value(j, R) ≤ X).
    out_lines = []
    for _ in range(q):
        try:
            R = int(next(it))
            X = int(next(it))
        except StopIteration:
            break
        # Lmax is the number of slots created by time R.
        Lmax = bisect_right(first_times, R)
        # Binary search for the count of slots in [0,Lmax) whose dp snapshot value ≤ X.
        lo = 0
        hi = Lmax
        while lo < hi:
            mid = (lo + hi) // 2
            if get_slot_value(mid, R) <= X:
                lo = mid + 1
            else:
                hi = mid
        ans = lo
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()