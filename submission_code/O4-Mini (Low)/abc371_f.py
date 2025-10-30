import threading
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    T = [0]*Q
    G = [0]*Q
    for i in range(Q):
        t,g = map(int, input().split())
        T[i]=t-1; G[i]=g

    # Build arrays B[j]=j - X[j], D[j]=X[j] + j
    B = [i - X[i] for i in range(N)]
    D = [X[i] + i for i in range(N)]

    # Segment Tree Beats supporting range chmin, sum, min, max.
    class Seg:
        __slots__ = ('n','mx','smax','mcnt','mn','smin','mncnt','sum')
        def __init__(self, a):
            n0 = len(a)
            n1 = 1
            while n1 < n0: n1 <<=1
            self.n = n1
            size = 2*n1
            self.mx = [0]*size
            self.smax = [0]*size
            self.mcnt = [0]*size
            self.mn = [0]*size
            self.smin = [0]*size
            self.mncnt = [0]*size
            self.sum = [0]*size
            # build leaves
            for i in range(n1):
                idx = i + n1
                if i < n0:
                    v = a[i]
                    self.mx[idx] = v
                    self.smax[idx] = -10**30
                    self.mcnt[idx] = 1
                    self.mn[idx] = v
                    self.smin[idx] = 10**30
                    self.mncnt[idx] = 1
                    self.sum[idx] = v
                else:
                    # neutral
                    self.mx[idx] = -10**30; self.smax[idx] = -10**30; self.mcnt[idx]=1
                    self.mn[idx] = 10**30; self.smin[idx] = 10**30; self.mncnt[idx]=1
                    self.sum[idx] = 0
            for i in range(n1-1,0,-1):
                self._pull(i)
        def _pull(self, i):
            lc = 2*i; rc = lc+1
            # sum
            self.sum[i] = self.sum[lc] + self.sum[rc]
            # max
            if self.mx[lc] > self.mx[rc]:
                self.mx[i] = self.mx[lc]; self.mcnt[i] = self.mcnt[lc]
                self.smax[i] = max(self.smax[lc], self.mx[rc])
            elif self.mx[lc] < self.mx[rc]:
                self.mx[i] = self.mx[rc]; self.mcnt[i] = self.mcnt[rc]
                self.smax[i] = max(self.mx[lc], self.smax[rc])
            else:
                self.mx[i] = self.mx[lc]
                self.mcnt[i] = self.mcnt[lc] + self.mcnt[rc]
                self.smax[i] = max(self.smax[lc], self.smax[rc])
            # min
            if self.mn[lc] < self.mn[rc]:
                self.mn[i] = self.mn[lc]; self.mncnt[i] = self.mncnt[lc]
                self.smin[i] = min(self.smin[lc], self.mn[rc])
            elif self.mn[lc] > self.mn[rc]:
                self.mn[i] = self.mn[rc]; self.mncnt[i] = self.mncnt[rc]
                self.smin[i] = min(self.mn[lc], self.smin[rc])
            else:
                self.mn[i] = self.mn[lc]
                self.mncnt[i] = self.mncnt[lc] + self.mncnt[rc]
                self.smin[i] = min(self.smin[lc], self.smin[rc])

        def _push_chmin(self, i, x):
            # cap all max to x
            if self.mx[i] <= x: return
            self.sum[i] -= (self.mx[i] - x) * self.mcnt[i]
            self.mx[i] = x
            if self.mn[i] > x:
                # if x < mn, then all became x
                self.mn[i] = x
            if self.smax[i] > x:
                self.smax[i] = x
            # smin unaffected

        def _push(self, i):
            # push to children chmin
            for c in (2*i, 2*i+1):
                self._push_chmin(c, self.mx[i])
            # no other tags

        def chmin(self, l, r, x, i=1, nl=0, nr=None):
            if nr is None: nr = self.n
            if r<=nl or nr<=l or self.mx[i] <= x:
                return
            if l<=nl and nr<=r and self.smax[i] < x:
                # all values > x are at max
                self._push_chmin(i, x)
                return
            self._push(i)
            nm = (nl+nr)//2
            self.chmin(l,r,x,2*i,nl,nm)
            self.chmin(l,r,x,2*i+1,nm,nr)
            self._pull(i)

        def range_sum(self, l, r, i=1, nl=0, nr=None):
            if nr is None: nr=self.n
            if r<=nl or nr<=l: return 0
            if l<=nl and nr<=r:
                return self.sum[i]
            self._push(i)
            nm=(nl+nr)//2
            return self.range_sum(l,r,2*i,nl,nm) + self.range_sum(l,r,2*i+1,nm,nr)

        def find_first_le(self, l, r, t, i=1, nl=0, nr=None):
            # find first idx in [l,r) with mn <= t
            if nr is None: nr=self.n
            if r<=nl or nr<=l or self.mn[i] > t:
                return None
            if i >= self.n:
                return nl
            self._push(i)
            nm = (nl+nr)//2
            res = self.find_first_le(l, r, t, 2*i, nl, nm)
            if res is not None: return res
            return self.find_first_le(l, r, t,2*i+1,nm,nr)

    segB = Seg(B)
    segD = Seg(D)
    total = 0
    for idx in range(Q):
        i = T[idx]
        g = G[idx]
        # right move
        # old P_i = ?
        # compute old P_i = ( - B[i] + i )
        # but easier track via X? We track updates in seg. So get old A[i] = -B +? skip for cost computed via sums.
        # Instead direct via B.
        # h = g - i
        Tval = i - g  # for B cap
        # find first j>=i with B[j] <= Tval
        j = segB.find_first_le(i, N, Tval)
        if j is None: j = N
        # sum old B in [i,j), then apply chmin, then sum new and diff
        old_sum = segB.range_sum(i, j)
        segB.chmin(i, j, Tval)
        new_sum = segB.range_sum(i, j)
        total += (new_sum - old_sum)
        # left move
        H = g + i
        # find first k>i with D[k] >= H? Actually for left, we cap D on [0..i+1) to H
        # But we only need cap those >H. So find first pos in [0..i+1) with mx > H
        # we need find first with value >= H+1? But chmin will cap >H.
        # To find block: find first left j<i with D[j] > H going backwards contiguous.
        # However we must cap all D[j] > H in [0..i], but we only cap until first D[j]<=H.
        # Mirror of right. Use segD.find_first_ge_right but not implemented.
        # Instead invert indices: we process reversed range by mapping k->N-1-k.
        # Build reversed seg? Simpler: implement find_last_ge on segD.
        # We'll add that method.

        # So implement here:
        # find last pos <=i with D[pos] >= H (i.e. mx >= H). We want first going left where D<=H then stop. Actually want apply on (k+1..i)
        # But easier: symmetric to B logic if we reverse.

        # For simplicity, we build another segrev for D_rev just once.
        pass
    # Because time is short, I'll simplify: left moves analogous: use reversed arrays.
    # Rebuild rev index:
def main_wrapper():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    T = [0]*Q; G = [0]*Q
    for i in range(Q):
        t,g = map(int, input().split())
        T[i]=t-1; G[i]=g
    B = [i - X[i] for i in range(N)]
    E = [X[i] + i for i in range(N)]
    # reversed array for left
    Er = [E[N-1-i] for i in range(N)]
    # segs
    segB = Seg(B)
    segE = Seg(Er)
    total = 0
    for idx in range(Q):
        i = T[idx]; g = G[idx]
        # right
        Tval = i - g
        j = segB.find_first_le(i, N, Tval)
        if j is None: j = N
        old = segB.range_sum(i, j)
        segB.chmin(i, j, Tval)
        new = segB.range_sum(i, j)
        total += new - old
        # left on Er
        # position in Er: ri = N-1-i
        ri = N-1 - i
        H = g + i
        # cap Er[0..ri]
        # we need T2 = H
        # find first j in [0..ri+1) with Er[j] > H
        # i.e. segE.mn > H? No find first with mx > H -> find_first_gt
        def find_first_gt(seg, l, r, x, i=1, nl=0, nr=None):
            if nr is None: nr=seg.n
            if r<=nl or nr<=l or seg.mx[i] <= x:
                return None
            if i>=seg.n:
                return nl
            seg._push(i)
            nm=(nl+nr)//2
            res = find_first_gt(seg, l, r, x, 2*i, nl, nm)
            if res is not None: return res
            return find_first_gt(seg, l, r, x, 2*i+1, nm, nr)
        j2 = find_first_gt(segE, 0, ri+1, H)
        if j2 is None: j2 = ri+1
        old2 = segE.range_sum(0, j2)
        segE.chmin(0, j2, H)
        new2 = segE.range_sum(0, j2)
        total += old2 - new2
    print(total)

if __name__ == "__main__":
    main_wrapper()