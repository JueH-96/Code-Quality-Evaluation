import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.readline
    line = data().split()
    if not line:
        return
    H = int(line[0]); W = int(line[1]); NB = int(line[2])
    bars = []
    for idx in range(NB):
        l = data().split()
        R = int(l[0]); C = int(l[1]); L = int(l[2])
        bars.append((R, C, L, idx))
    # process bars in descending initial row
    bars.sort(key=lambda x: -x[0])
    # segment tree over columns [0..W-1]
    N0 = 1
    while N0 < W:
        N0 <<= 1
    INF = H + 1
    size = N0 << 1
    minv = [INF] * size
    lazy = [None] * size

    def push(k):
        """ Push down the lazy assign from node k to its children. """
        v = lazy[k]
        if v is not None:
            lc = k << 1; rc = lc | 1
            minv[lc] = v; minv[rc] = v
            lazy[lc] = v; lazy[rc] = v
            lazy[k] = None

    def pull(k):
        """ Recompute minv[k] from its children. """
        lc = k << 1; rc = lc | 1
        mv = minv[lc]
        rv = minv[rc]
        # inline min
        minv[k] = mv if mv < rv else rv

    def update(a, b, v, k=1, l=0, r=N0-1):
        """ Assign the value v to all positions in [a..b]. """
        if b < l or r < a:
            return
        if a <= l and r <= b:
            minv[k] = v
            lazy[k] = v
            return
        # partial cover
        push(k)
        m = (l + r) // 2
        nk = k << 1
        if a <= m:
            update(a, b, v, nk, l, m)
        if b > m:
            update(a, b, v, nk | 1, m+1, r)
        pull(k)

    def query(a, b, k=1, l=0, r=N0-1):
        """ Return min over positions in [a..b]. """
        if b < l or r < a:
            return INF
        if a <= l and r <= b:
            return minv[k]
        push(k)
        m = (l + r) // 2
        mv = INF
        if a <= m:
            mv = query(a, b, k<<1, l, m)
        if b > m:
            rv = query(a, b, k<<1|1, m+1, r)
            if rv < mv:
                mv = rv
        return mv

    ans = [0] * NB
    # local references for speed
    upd = update
    qry = query

    for R, C, L, idx in bars:
        lcol = C - 1
        rcol = C + L - 2
        mf = qry(lcol, rcol)
        newR = mf - 1
        ans[idx] = newR
        upd(lcol, rcol, newR)

    out = sys.stdout.write
    for v in ans:
        out(str(v))
        out("
")

if __name__ == "__main__":
    main()