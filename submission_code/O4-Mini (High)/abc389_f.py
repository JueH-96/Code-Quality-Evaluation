import sys
import threading

def main():
    import sys
    data = sys.stdin
    readline = data.readline

    # Read input
    N = int(readline())
    L = [0] * N
    R = [0] * N
    for i in range(N):
        l, r = map(int, readline().split())
        L[i] = l
        R[i] = r
    Q = int(readline())
    queries = [0] * Q
    for i in range(Q):
        queries[i] = int(readline())

    # Coordinate-compress the query X values
    uniq = sorted(set(queries))
    K = len(uniq)
    idx_map = {v: i for i, v in enumerate(uniq)}

    # Build a segment tree over the K query positions
    # We'll keep a power-of-two size
    size = 1 << (K - 1).bit_length()
    NEG_INF = -(10**18)
    # t[p] = current max f in the node p
    # add[p] = lazy add to propagate
    t = [NEG_INF] * (2 * size)
    add = [0] * (2 * size)

    # Initialize leaves with the initial f(X) = X for each query X
    base = size
    for i, xval in enumerate(uniq):
        t[base + i] = xval
    # Leaves beyond K stay NEG_INF so they never satisfy >= conditions

    # Build internal nodes
    for p in range(size - 1, 0, -1):
        # max of children
        lc = p << 1
        rc = lc | 1
        t[p] = t[lc] if t[lc] > t[rc] else t[rc]

    # Function to find leftmost index j in [0..K-1] with f[j] >= val
    # Returns K if none.
    def find_first_ge(val):
        if t[1] < val:
            return K
        p = 1
        l0 = 0
        r0 = size - 1
        t_arr = t
        add_arr = add
        while l0 != r0:
            # push-down lazy
            a = add_arr[p]
            if a:
                lc = p << 1
                rc = lc | 1
                t_arr[lc] += a
                add_arr[lc] += a
                t_arr[rc] += a
                add_arr[rc] += a
                add_arr[p] = 0
            mid = (l0 + r0) >> 1
            lc = p << 1
            # go left if left child can satisfy
            if t_arr[lc] >= val:
                p = lc
                r0 = mid
            else:
                p = lc | 1
                l0 = mid + 1
        return l0

    # Range-add v to [l..r] (0-indexed on compressed coords)
    def add_range(l, r, v, p=1, lp=0, rp=None):
        if rp is None:
            rp = size - 1
        if r < lp or rp < l:
            return
        if l <= lp and rp <= r:
            t[p] += v
            add[p] += v
            return
        # push-down
        a = add[p]
        if a:
            lc = p << 1
            rc = lc | 1
            t[lc] += a
            add[lc] += a
            t[rc] += a
            add[rc] += a
            add[p] = 0
        mid = (lp + rp) >> 1
        c = p << 1
        add_range(l, r, v, c,     lp,   mid)
        add_range(l, r, v, c | 1, mid+1, rp)
        # pull-up
        tl = t[c]
        tr = t[c | 1]
        t[p] = tl if tl > tr else tr

    # Point-get f[j] for compressed index j
    def get_idx(j):
        p = 1
        l0 = 0
        r0 = size - 1
        t_arr = t
        add_arr = add
        while l0 != r0:
            a = add_arr[p]
            if a:
                lc = p << 1
                rc = lc | 1
                t_arr[lc] += a
                add_arr[lc] += a
                t_arr[rc] += a
                add_arr[rc] += a
                add_arr[p] = 0
            mid = (l0 + r0) >> 1
            if j <= mid:
                p = p << 1
                r0 = mid
            else:
                p = (p << 1) | 1
                l0 = mid + 1
        return t[p]

    # Process each contest in order, computing the preimage on query X's
    for i in range(N):
        Li = L[i]
        Ri = R[i]
        # leftmost query index with f >= Li
        l = find_first_ge(Li)
        if l >= K:
            continue
        # rightmost with f <= Ri:
        # find first f >= Ri+1, then r = that - 1
        rr = find_first_ge(Ri + 1)
        if rr >= K:
            r = K - 1
        else:
            r = rr - 1
        if l <= r:
            # those query indices get +1 to f
            add_range(l, r, 1)

    # Answer each query
    out = []
    wm = out.append
    for x in queries:
        j = idx_map[x]
        wm(str(get_idx(j)))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()