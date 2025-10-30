import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    M = int(input())
    A = list(map(int, input().split()))
    # We maintain a set S of all i with P[i] > P[i+1], 0-based.
    # Each adjacent-swap reduces inversion count by exactly 1.
    # Each operation k does one bubble‐pass over [0..k-1]:
    #   for i in 0..k-2 in order, if (i in S) then swap P[i],P[i+1].
    # We simulate only the swaps by jumping from one inversion to the next via a sorted list.
    import bisect

    # Build initial inversion‐set
    S = []
    for i in range(N-1):
        if P[i] > P[i+1]:
            S.append(i)
    S.sort()
    # Initial inversion count via BIT
    class BIT:
        def __init__(self,n):
            self.n=n
            self.f=[0]*(n+1)
        def add(self,i,v):
            while i<=self.n:
                self.f[i]+=v; i+=i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.f[i]; i-=i&-i
            return s
    bit = BIT(N)
    inv = 0
    for i,x in enumerate(P):
        # count how many > x seen before
        inv += i - bit.sum(x)
        bit.add(x,1)

    out = []
    # We'll maintain a pointer for each operation: we need to process
    # all inversion‐positions i < k-1 in increasing i, but new inversions
    # created by earlier swaps can be for positions < i so won't affect
    # this pass.  The known trick: one bubble‐pass just removes each
    # inversion-edge once.  Hence we can greedily take all S-elements < k-1
    # in sorted order and swap them exactly once.
    for k in A:
        limit = k-1  # we handle S[i] < limit
        # find all in S less than limit
        idx = bisect.bisect_left(S, limit)
        to_do = S[:idx]
        S = S[idx:]
        # perform these swaps in ascending order of positions
        for i in to_do:
            # swap P[i],P[i+1]
            inv -= 1
            x, y = P[i], P[i+1]
            P[i], P[i+1] = y, x
            # positions i-1, i, i+1 might change inversion‐status
            for j in (i-1, i, i+1):
                if 0 <= j < N-1:
                    # remove old if present
                    pos = bisect.bisect_left(S, j)
                    if pos < len(S) and S[pos]==j:
                        S.pop(pos)
                    # re-test
                    if P[j] > P[j+1]:
                        bisect.insort(S, j)
        out.append(str(inv))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()