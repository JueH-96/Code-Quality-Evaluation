from bisect import bisect_left as bl, bisect_right as br, insort

class SegTree:
    def __init__(self, a):
        n = len(a)
        self.a = [0] * (4 * n)
        self.build(a, 0, 0, n)

    def build(self, a, i, l, r):
        if r - l == 1:
            if l < len(a):
                self.a[i] = a[l]
            return
        mid = l + (r - l) // 2
        self.build(a, 2 * i + 1, l, mid)
        self.build(a, 2 * i + 2, mid, r)
        self.a[i] = (self.a[2 * i + 1] if self.a[2 * i + 1] > 0 else 0) + (self.a[2 * i + 2] if self.a[2 * i + 2] > 0 else 0)

    def query(self, i, l, r, ll, rr):
        if l >= rr or r <= ll:
            return 0
        if ll <= l and r <= rr:
            return self.a[i] if self.a[i] > 0 else 0
        mid = l + (r - l) // 2
        return self.query(2 * i + 1, l, mid, ll, rr) + self.query(2 * i + 2, mid, r, ll, rr)

    def update(self, i, l, r, pos, val):
        if r - l == 1:
            self.a[i] = val
            return
        mid = l + (r - l) // 2
        if pos < mid:
            self.update(2 * i + 1, l, mid, pos, val)
        else:
            self.update(2 * i + 2, mid, r, pos, val)
        if self.a[2 * i + 1] > 0:
            self.a[i] = self.a[2 * i + 1] + (self.a[2 * i + 2] if self.a[2 * i + 2] > 0 else 0)
        else:
            self.a[i] = self.a[2 * i + 2]

    def get(self, i):
        return self.query(0, 0, len(self.a), i, i + 1)

    def update_pos(self, pos, val):
        self.update(0, 0, len(self.a), pos, val)

def solve(N, M, X, A):
    pos = sorted(zip(X, A))
    cs = functools.reduce(lambda a, b: a + [a[-1] + b[1]], pos, [0])
    Xi = [x[0] - 1 for x in pos]
    Ai = [x[1] for x in pos]
    sep = min(M, N - M + 1)
    L, R = Xi[:sep], Xi[sep:]
    Ai = Ai[:sep]
    segtree = SegTree(Ai)
    insort(L, 0)
    Niouses = -1

    def dyn_prog(l, r):
        nonlocal Niouses
        if Niouses > -1:
            return
        if l == sep and r == 0:
            if (nstones := sum(Ai)) == N and nstones == N:
                Niouses = cs[sep - 1]
            return
        if r > 0:
            nstones = cs[sep - 1] - cs[r - 1] + Ai[r] * r
            cost = Ai[r] * (r - 1)
            if nstones + M - sep + r == N:
                Niouses = min(Niouses, cost) if Niouses > -1 else cost
            dyn_prog(l, r - 1)
            if l == sep or br(L, L[l] + r - L[l]) > r:
                insort(L, L[l] + r)
                Ai[l] += 1
                Ai[r] -= 1
                segtree.update_pos(l, Ai[l])
                segtree.update_pos(r, Ai[r])
                l += 1
                dyn_prog(l, r)
                Ai[l] -= 1
                Ai[r] += 1
                segtree.update_pos(l, Ai[l])
                segtree.update_pos(r, Ai[r])
                l -= 1
                L.pop(br(L, L[l] + r - L[l]) - 1)
            return
        if l < sep:
            nstones = sum(Ai) + Ai[l]
            cost = Ai[l]
            if nstones == N:
                Niouses = min(Niouses, cost) if Niouses > -1 else cost
            dyn_prog(l + 1, 0)
            if r == 0 or bl(R, R[r] + L[l] - R[r]) < l:
                Ai[l] += 1
                Ai[r] -= 1
                segtree.update_pos(l, Ai[l])
                segtree.update_pos(r, Ai[r])
                insort(R, L[l])
                r += 1
                dyn_prog(l, r)
                Ai[l] -= 1
                Ai[r] -= 1
                segtree.update_pos(l, Ai[l])
                segtree.update_pos(r, Ai[r])
                R.pop(br(R, L[l] - (R[r] - L[l])) - 1)
                r -= 1

    dyn_prog(0, 0)
    ans = float("inf") if Niouses == -1 else Niouses
    segtree = SegTree(Ai)
    for i in range(sep):
        if Xi[i] < N - M + 1:
            ans += cs[i] + (N - M) * Xi[i]
        else:
            li = bisect_left(L, Xi[i]) - 1
            ri = bisect_left(R, Xi[i])
            sm = sum(Ai[:li]) + segtree.query(0, 0, len(Ai), li + 1, ri)
            ans += cs[li] + sm + (N - M + 1 - (li + 1)) * Xi[i]
    return -1 if ans == float("inf") else ans

# print(solve(3, 2, [2, 3], [1, 1]))
# print(solve(4, 2, [1, 4], [1, 1]))
print(solve(5, 2, [1, 4], [3, 2]))
print(solve(10, 3, [1, 4, 8], [4, 2, 4]))