import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = int(data[2])
    B = int(data[3])
    bad = []
    idx = 4
    for _ in range(M):
        l = int(data[idx]); r = int(data[idx+1])
        idx += 2
        bad.append((l, r))
    # build good intervals
    good = []
    prev_end = 1
    for (l, r) in bad:
        if prev_end <= l-1:
            good.append((prev_end, l-1))
        prev_end = r+1
    if prev_end <= N:
        good.append((prev_end, N))
    # helper: merge sorted, non-overlapping segments
    def merge_segs(segs):
        if not segs:
            return []
        segs.sort()
        merged = []
        cur_l, cur_r = segs[0]
        for l, r in segs[1:]:
            if l <= cur_r+1:
                if r > cur_r: cur_r = r
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))
        return merged
    # helper: intersect segs with good intervals
    def intersect_with_good(segs):
        # segs and good both sorted
        res = []
        i = 0
        G = good
        for l, r in segs:
            # advance G until potential overlap
            while i < len(G) and G[i][1] < l:
                i += 1
            j = i
            while j < len(G) and G[j][0] <= r:
                gl, gr = G[j]
                il = max(l, gl)
                ir = min(r, gr)
                if il <= ir:
                    res.append((il, ir))
                j += 1
        return res
    # BFS-like on segments
    # seen holds merged reachable segments
    seen = []
    # initial reachable is square 1
    seen = [(1,1)]
    # queue of new segments to expand
    queue = [(1,1)]
    seen = merge_segs(seen)
    # quick check if N==1
    if N == 1:
        print("Yes")
        return
    while queue:
        # build next-level candidate jumps
        nxt = []
        for l, r in queue:
            nl = l + A
            nr = r + B
            if nl > N:
                continue
            if nr > N:
                nr = N
            nxt.append((nl, nr))
        # merge and intersect with good
        nxt = merge_segs(nxt)
        nxt = intersect_with_good(nxt)
        if not nxt:
            break
        # clip against already seen, and form new queue
        new_q = []
        # merge seen first
        seen = merge_segs(seen)
        # for each segment in nxt, subtract seen parts
        si = 0
        for l, r in nxt:
            cur = l
            # advance si while seen[si].r < cur
            while si < len(seen) and seen[si][1] < cur:
                si += 1
            # now either no more seen or seen[si].r >= cur
            while si < len(seen) and seen[si][0] <= r:
                sl, sr = seen[si]
                if sl > cur:
                    # [cur, sl-1] is new
                    new_q.append((cur, sl-1))
                cur = max(cur, sr+1)
                if cur > r:
                    break
                si += 1
            if cur <= r:
                new_q.append((cur, r))
        if not new_q:
            break
        # add new_q to seen
        seen += new_q
        seen = merge_segs(seen)
        queue = new_q
        # check if N seen
        # since seen sorted, quick check
        for l, r in new_q:
            if l <= N <= r:
                print("Yes")
                return
    # after BFS, check if N in seen
    for l, r in seen:
        if l <= N <= r:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()