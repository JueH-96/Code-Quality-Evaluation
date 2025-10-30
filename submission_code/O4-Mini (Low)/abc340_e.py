import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    # Fenwick for range add, point query
    class Fenw:
        def __init__(self,n):
            self.n = n
            self.fw = [0]*(n+1)
        def add(self,i,v):
            # add v at index i (0-based)
            i += 1
            n = self.n
            fw = self.fw
            while i <= n:
                fw[i] += v
                i += i & -i
        def prefix(self,i):
            # sum [0..i]
            s = 0
            i += 1
            fw = self.fw
            while i>0:
                s += fw[i]
                i -= i & -i
            return s
        def range_add(self,l,r,v):
            # add v to [l..r]
            if l<=r:
                self.add(l, v)
                if r+1 < self.n:
                    self.add(r+1, -v)
            else:
                # wrap
                self.add(l, v)
                # end
                # [l..n-1]
                # and [0..r]
                if N>0:
                    # end wrap
                    # no need r+1 check for r<n-1 always
                    if 0 <= r:
                        self.add(0, v)
                        if r+1 < self.n:
                            self.add(r+1, -v)
                # and for [l..N-1], we only did self.add(l,v) so need no negative at N since prefix end
                # but fenw beyond n is ignored
                # Actually, for [l..n-1], adding at l is enough; no end marker
                pass

    bit = Fenw(N)
    for b in B:
        # get current count at box b
        cur = A[b] + bit.prefix(b)
        if cur == 0:
            continue
        # full loops
        full = cur // N
        rem = cur - full * N
        if full:
            # add full to all
            bit.range_add(0, N-1, full)
        if rem:
            # add 1 to rem positions starting at (b+1)%N
            l = b+1
            if l >= N:
                l -= N
            # r = (b + rem) % N
            r = b + rem
            # but note rem < N so r < b+N; mod N:
            if r >= N:
                r -= N
            bit.range_add(l, r, 1)
        # remove all from b
        bit.range_add(b, b, -cur)
    # build result
    out = []
    for i in range(N):
        out.append(str(A[i] + bit.prefix(i)))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()