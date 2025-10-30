def main():
    import sys, bisect
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = [int(next(it)) for _ in range(n)]
    # Build initial A: A[i] = X[i] - i.
    A = [x - i for i, x in enumerate(X)]
    segments = []
    l = 0
    for i in range(1, n):
        if A[i] != A[i-1]:
            segments.append((l, i, A[i-1]))
            l = i
    segments.append((l, n, A[n-1]))
    seg_starts = [seg[0] for seg in segments]
    
    def insert_segment(seg):
        nonlocal segments, seg_starts
        l, r, val = seg
        pos = bisect.bisect_left(seg_starts, l)
        segments.insert(pos, seg)
        seg_starts.insert(pos, l)
        if pos > 0:
            pl, pr, pval = segments[pos-1]
            if pval == val and pr == l:
                new_seg = (pl, r, val)
                segments[pos-1] = new_seg
                seg_starts[pos-1] = pl
                del segments[pos]
                del seg_starts[pos]
                pos -= 1
                l, r, val = new_seg
        if pos+1 < len(segments):
            nl, nr, nval = segments[pos+1]
            if nval == val and r == nl:
                new_seg = (l, nr, val)
                segments[pos] = new_seg
                seg_starts[pos] = l
                del segments[pos+1]
                del seg_starts[pos+1]
        return

    def remove_segment_at(pos):
        nonlocal segments, seg_starts
        del segments[pos]
        del seg_starts[pos]
    
    def split_segment(seg_index, x):
        nonlocal segments, seg_starts
        l, r, val = segments[seg_index]
        leftseg = (l, x, val)
        rightseg = (x, r, val)
        segments[seg_index] = leftseg
        seg_starts[seg_index] = l
        segments.insert(seg_index+1, rightseg)
        seg_starts.insert(seg_index+1, x)
        return

    def find_segment(x):
        nonlocal seg_starts, segments
        pos = bisect.bisect_right(seg_starts, x) - 1
        return pos

    q = int(next(it))
    total_cost = 0
    for _ in range(q):
        T = int(next(it))
        G = int(next(it))
        t = T - 1
        K = G - t
        seg_index = find_segment(t)
        l, r, val = segments[seg_index]
        if l < t:
            split_segment(seg_index, t)
            seg_index += 1
            l, r, val = segments[seg_index]
        if t+1 < r:
            split_segment(seg_index, t+1)
            l, r, val = segments[seg_index]
        total_cost += abs(val - K)
        segments[seg_index] = (t, t+1, K)
        seg_starts[seg_index] = t
        cur_l = t
        while seg_index > 0:
            prev_seg_index = seg_index - 1
            pl, pr, pval = segments[prev_seg_index]
            if pr != cur_l:
                break
            if pval > K:
                length = pr - pl
                total_cost += (pval - K) * length
                cur_l = pl
                remove_segment_at(prev_seg_index)
                seg_index -= 1
                segments[seg_index] = (cur_l, segments[seg_index][1], K)
                seg_starts[seg_index] = cur_l
            else:
                break
        cur_r = t+1
        while seg_index < len(segments)-1:
            next_seg_index = seg_index + 1
            nl, nr, nval = segments[next_seg_index]
            if nl != cur_r:
                break
            if nval < K:
                length = nr - nl
                total_cost += (K - nval) * length
                cur_r = nr
                remove_segment_at(next_seg_index)
                segments[seg_index] = (segments[seg_index][0], cur_r, K)
                seg_starts[seg_index] = segments[seg_index][0]
            else:
                break
        # (Optional merging with neighbors is already handled by our removal/merging process.)
    sys.stdout.write(str(total_cost))
    
if __name__ == '__main__':
    main()