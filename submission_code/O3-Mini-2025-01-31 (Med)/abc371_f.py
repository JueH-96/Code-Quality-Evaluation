# YOUR CODE HERE
def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read positions A[1..n]
    A = [0]*(n+1)
    for i in range(1, n+1):
        A[i] = int(next(it))
    q = int(next(it))
    
    # Initially, B[i] = A[i] - i. (For 1-indexed i.)
    # Note that A is strictly increasing so B is non-decreasing.
    B = [0]*(n+1)
    for i in range(1, n+1):
        B[i] = A[i] - i

    # Compress B into segments.
    # Each segment is represented as [l, r, val] meaning that for all i in [l,r], B[i] = val.
    segs = []
    l = 1
    cur = B[1]
    for i in range(2, n+1):
        if B[i] != cur:
            segs.append([l, i-1, cur])
            l = i
            cur = B[i]
    segs.append([l, n, cur])
    
    total_cost = 0

    # Helper: given an index t, find the segment (its index in segs) which contains t.
    def find_seg(t):
        # segments are sorted by their starting index.
        pos = bisect.bisect_right(segs, [t, 10**18, 10**18]) - 1
        return pos

    # Process each query.
    # For a query (T, G), let C = G - T.
    # Then update:
    #  - for i < T, if B[i] > C then set it to C.
    #  - for i = T, set B[T] = C.
    #  - for i > T, if B[i] < C then set it to C.
    # And the cost at i is the change |oldB[i] - C|.
    # We update our segs (which represent B) accordingly.
    for _ in range(q):
        t = int(next(it))
        G = int(next(it))
        C = G - t  # new forced value for B[t].
        
        # 1. Find the segment containing t.
        idx = find_seg(t)
        seg = segs[idx]
        old_val = seg[2]
        total_cost += abs(old_val - C)
        
        # Split the segment containing t into up-to three parts.
        new_segments = []
        if seg[0] < t:
            new_segments.append([seg[0], t-1, seg[2]])
        # mid segment: just t gets the new value C.
        mid_seg = [t, t, C]
        new_segments.append(mid_seg)
        if t < seg[1]:
            new_segments.append([t+1, seg[1], seg[2]])
        
        segs[idx:idx+1] = new_segments
        # Now determine where mid_seg is located.
        if new_segments[0][0] < t:
            anchor_idx = idx + 1
        else:
            anchor_idx = idx
        
        # Merge left: while the immediate left segment has value > C, update it.
        while anchor_idx - 1 >= 0:
            left_seg = segs[anchor_idx - 1]
            if left_seg[2] > C:
                length = left_seg[1] - left_seg[0] + 1
                total_cost += (left_seg[2] - C) * length
                new_left = left_seg[0]
                del segs[anchor_idx - 1]
                anchor_idx -= 1
                mid_seg[0] = new_left
            elif left_seg[2] == C:
                new_left = left_seg[0]
                del segs[anchor_idx - 1]
                anchor_idx -= 1
                mid_seg[0] = new_left
            else:
                break
        
        # Merge right: while the immediate right segment has value < C, update it.
        while anchor_idx + 1 < len(segs):
            right_seg = segs[anchor_idx + 1]
            if right_seg[2] < C:
                length = right_seg[1] - right_seg[0] + 1
                total_cost += (C - right_seg[2]) * length
                new_right = right_seg[1]
                del segs[anchor_idx + 1]
                mid_seg[1] = new_right
            elif right_seg[2] == C:
                new_right = right_seg[1]
                del segs[anchor_idx + 1]
                mid_seg[1] = new_right
            else:
                break

    sys.stdout.write(str(total_cost))

if __name__ == '__main__':
    main()