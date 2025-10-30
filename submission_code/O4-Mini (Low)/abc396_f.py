import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1])
    A = list(map(int, data[2:]))

    # Fenwick / BIT for initial inversion count over range [0..M-1]
    class Fenwick:
        def __init__(self,n):
            self.n = n
            self.f = [0]*(n+1)
        def add(self,i,v):
            i += 1
            while i <= self.n:
                self.f[i] += v
                i += i & -i
        def sum(self,i):
            # sum of [0..i)
            s = 0
            while i>0:
                s += self.f[i]
                i -= i & -i
            return s
        def range_sum(self,l,r):
            return self.sum(r) - self.sum(l)

    # compute inv0
    bit = Fenwick(M)
    inv0 = 0
    for i,x in enumerate(A):
        # how many have come before that are > x?
        inv0 += i - bit.sum(x+1)
        bit.add(x,1)

    # prepare cnt[r] and sum_pos[r]
    cnt = [0]*M
    sum_pos = [0]*M
    for i,x in enumerate(A, start=1):  # positions are 1-based in formula
        cnt[x] += 1
        sum_pos[x] += i

    # output
    out = []
    inv = inv0
    out.append(str(inv))
    # for each k = 1..M-1
    # the residue that wraps is r = (M-k)%M
    # delta = 2*sum_pos[r] - cnt[r] - cnt[r]*N
    for k in range(1, M):
        r = (M - k) % M
        c = cnt[r]
        s = sum_pos[r]
        delta = 2*s - c - c*N
        inv += delta
        out.append(str(inv))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()