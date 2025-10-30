import sys
sys.setrecursionlimit(10**7)
def main():
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    # initial inversion count via Fenwick (BIT)
    class Fenwick:
        def __init__(self,n):
            self.n=n
            self.fw=[0]*(n+1)
        def add(self,i,v):
            # i in [1..n]
            while i<=self.n:
                self.fw[i]+=v
                i+=i&-i
        def sum(self,i):
            # sum 1..i
            s=0
            while i>0:
                s+=self.fw[i]
                i-=i&-i
            return s
    fw = Fenwick(N)
    inv = 0
    for i,x in enumerate(P):
        # how many previous > x ?
        gt = i - fw.sum(x)
        inv += gt
        fw.add(x,1)
    # segment tree over positions 0..N-2 storing 1 if P[i]>P[i+1]
    size = 1
    while size < N-1: size <<= 1
    st = [0]*(size<<1)
    def st_set(pos,val):
        # pos in [0..N-2]
        i = pos + size
        st[i] = val
        i >>= 1
        while i:
            st[i] = 1 if (st[i<<1] or st[(i<<1)|1]) else 0
            i >>= 1
    def st_find_first(node,nl,nr,ql,qr):
        # find leftmost position in [ql..qr] with st[...] == 1
        if ql>nr or qr<nl or st[node]==0:
            return -1
        if nl==nr:
            return nl
        mid = (nl+nr)>>1
        left = st_find_first(node<<1, nl, mid, ql, qr)
        if left != -1:
            return left
        return st_find_first((node<<1)|1, mid+1, nr, ql, qr)
    # build initial descent set
    for i in range(N-1):
        st_set(i, 1 if P[i] > P[i+1] else 0)
    M = int(input())
    A = list(map(int, input().split()))
    out = []
    for k in A:
        # one bubble‑sort pass on prefix length k
        # i.e.\ comparators at (0,1),(1,2),...,(k-2,k-1)
        cur = 0
        swaps = 0
        # we look for the next descent position >=cur and <k-1
        while True:
            pos = st_find_first(1, 0, size-1, cur, k-2)
            if pos < 0 or pos >= k-1:
                break
            # swap P[pos], P[pos+1]
            P[pos], P[pos+1] = P[pos+1], P[pos]
            swaps += 1
            inv -= 1  # each swap removes exactly one inversion
            # update descents at pos-1, pos, pos+1
            for d in (pos-1, pos, pos+1):
                if 0 <= d < N-1:
                    st_set(d, 1 if P[d] > P[d+1] else 0)
            cur = pos + 1
        out.append(str(inv))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()