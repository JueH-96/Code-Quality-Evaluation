import threading
import sys
import threading
def main():
    import sys
    import threading
    sys.setrecursionlimit(1 << 25)
    from bisect import bisect_right

    input = sys.stdin.readline
    N, M, Q = map(int, input().split())
    L = [0] * (M + 1)
    R = [0] * (M + 1)
    d = [0] * (M + 1)
    for i in range(1, M + 1):
        s, t = map(int, input().split())
        if s < t:
            L[i], R[i], d[i] = s, t, 1
        else:
            L[i], R[i], d[i] = t, s, -1

    # coordinate compress L[i]
    coords = sorted(set(L[1:]))
    K = len(coords)
    # map l value to idx 0-based
    comp = {v: i for i, v in enumerate(coords)}

    # Segment tree for max, 0-based, length K
    class SegTree:
        def __init__(self, n):
            self.n0 = 1
            while self.n0 < n:
                self.n0 <<= 1
            size = 2 * self.n0
            self.data = [0] * size

        def set_val(self, pos, val):
            i = pos + self.n0
            if self.data[i] == val:
                return
            self.data[i] = val
            i >>= 1
            while i:
                nv = self.data[2 * i]
                rv = self.data[2 * i + 1]
                if nv >= rv:
                    self.data[i] = nv
                else:
                    self.data[i] = rv
                i >>= 1

        # query max in [l, r] inclusive, 0-based
        def query(self, l, r):
            if r < l:
                return 0
            l += self.n0
            r += self.n0
            res = 0
            while l <= r:
                if (l & 1) == 1:
                    if self.data[l] > res: res = self.data[l]
                    l += 1
                if (r & 1) == 0:
                    if self.data[r] > res: res = self.data[r]
                    r -= 1
                l >>= 1; r >>= 1
            return res

    # Buckets for fwd and bwd
    import heapq
    # Each bucket has main heap and del heap for lazy deletion
    fwd_main = [[] for _ in range(K)]
    fwd_del = [[] for _ in range(K)]
    bwd_main = [[] for _ in range(K)]
    bwd_del = [[] for _ in range(K)]
    # Segment trees
    st_fwd = SegTree(K)
    st_bwd = SegTree(K)

    # helper to clean bucket at idx for given direction and return current max
    def bucket_max(idx, is_fwd):
        if is_fwd:
            main = fwd_main[idx]; dele = fwd_del[idx]
        else:
            main = bwd_main[idx]; dele = bwd_del[idx]
        # clean lazy
        while main and dele and main[0] == dele[0]:
            heapq.heappop(main); heapq.heappop(dele)
        if not main:
            return 0
        # main stores -r, so max r = -main[0]
        return -main[0]

    next_bad = [M + 1] * (M + 2)
    rptr = 0
    # sliding window l..rptr, 1-based indices
    for lptr in range(1, M + 1):
        # extend rptr
        while rptr + 1 <= M:
            i = rptr + 1
            l0 = L[i]; r0 = R[i]; di = d[i]
            a = l0
            b = r0 - 2
            # find pos limit for l_i <= b
            # coords[idx] <= b
            idx_b = bisect_right(coords, b) - 1
            conflict = False
            if idx_b >= 0:
                if di == 1:
                    # check any backward interval overlapping
                    max_rj = st_bwd.query(0, idx_b)
                else:
                    max_rj = st_fwd.query(0, idx_b)
                # overlapping condition: r_j >= a+2
                if max_rj >= a + 2:
                    conflict = True
            if conflict:
                break
            # no conflict -> insert interval i
            idx0 = comp[l0]
            if di == 1:
                # forward insert
                heapq.heappush(fwd_main[idx0], -r0)
                # update segtree with new bucket max
                mv = bucket_max(idx0, True)
                st_fwd.set_val(idx0, mv)
            else:
                # backward insert
                heapq.heappush(bwd_main[idx0], -r0)
                mv = bucket_max(idx0, False)
                st_bwd.set_val(idx0, mv)
            rptr += 1
        # record next_bad for lptr
        next_bad[lptr] = rptr + 1
        # remove interval lptr if in window
        if lptr <= rptr:
            i = lptr
            l0 = L[i]; r0 = R[i]; di = d[i]
            idx0 = comp[l0]
            if di == 1:
                # forward removal
                heapq.heappush(fwd_del[idx0], -r0)
                mv = bucket_max(idx0, True)
                st_fwd.set_val(idx0, mv)
            else:
                heapq.heappush(bwd_del[idx0], -r0)
                mv = bucket_max(idx0, False)
                st_bwd.set_val(idx0, mv)
        # else window empty, nothing to remove

    # answer queries
    out = []
    for _ in range(Q):
        Lq, Rq = map(int, input().split())
        if Rq < next_bad[Lq]:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()