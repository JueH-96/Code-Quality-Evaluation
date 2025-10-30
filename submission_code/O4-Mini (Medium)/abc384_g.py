import threading
import sys
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    K = int(input())
    queries = []
    for idx in range(K):
        x, y = map(int, input().split())
        queries.append((x, y, idx))
    # Coordinate compress A and B values
    vals = list(set(A) | set(B))
    vals.sort()
    comp = {v:i+1 for i,v in enumerate(vals)}  # 1-indexed
    M = len(vals) + 2
    A_comp = [comp[v] for v in A]
    B_comp = [comp[v] for v in B]
    # Fenwick Tree
    class BIT:
        __slots__ = ('n','t')
        def __init__(self,n):
            self.n = n
            self.t = [0]*(n+1)
        def add(self,i,v):
            while i <= self.n:
                self.t[i] += v
                i += i & -i
        def sum(self,i):
            s = 0
            t = self.t
            while i > 0:
                s += t[i]
                i -= i & -i
            return s
        def range_sum(self,l,r):
            if r<l: return 0
            return self.sum(r) - self.sum(l-1)
    # Two BITs for A and B: counts and sums
    BIT_A_cnt = BIT(M)
    BIT_A_sum = BIT(M)
    BIT_B_cnt = BIT(M)
    BIT_B_sum = BIT(M)
    # Mo's ordering on (X,Y)
    # block size on X
    import math
    Bsize = int(math.sqrt(N)) or 1
    # sort queries by block of x, then y (odd even trick)
    queries.sort(key=lambda q: (q[0]//Bsize, q[1] if ((q[0]//Bsize)&1)==0 else -q[1]))
    curX = 0
    curY = 0
    sumA = 0  # sum of current A[1..curX]
    sumB = 0  # sum of current B[1..curY]
    curAns = 0
    res = [0]*K
    for X, Y, qi in queries:
        # expand/shrink X
        while curX < X:
            # add A[curX]
            v = A[curX]
            c = A_comp[curX]
            # contribution: sum over j<=curY of |v - B_j|
            cnt_le = BIT_B_cnt.sum(c)
            sum_le = BIT_B_sum.sum(c)
            # cnt_gt = curY - cnt_le, sum_gt = sumB - sum_le
            curAns += v*cnt_le - sum_le + (sumB - sum_le) - v*(curY - cnt_le)
            # add to A trees
            BIT_A_cnt.add(c, 1)
            BIT_A_sum.add(c, v)
            sumA += v
            curX += 1
        while curX > X:
            # remove A[curX-1]
            curX -= 1
            v = A[curX]
            c = A_comp[curX]
            # remove its contribution
            cnt_le = BIT_B_cnt.sum(c)
            sum_le = BIT_B_sum.sum(c)
            # but B hasn't changed; curY same
            curAns -= v*cnt_le - sum_le + (sumB - sum_le) - v*(curY - cnt_le)
            BIT_A_cnt.add(c, -1)
            BIT_A_sum.add(c, -v)
            sumA -= v
        # expand/shrink Y
        while curY < Y:
            # add B[curY]
            v = B[curY]
            c = B_comp[curY]
            cnt_le = BIT_A_cnt.sum(c)
            sum_le = BIT_A_sum.sum(c)
            curAns += v*cnt_le - sum_le + (sumA - sum_le) - v*(curX - cnt_le)
            BIT_B_cnt.add(c, 1)
            BIT_B_sum.add(c, v)
            sumB += v
            curY += 1
        while curY > Y:
            # remove B[curY-1]
            curY -= 1
            v = B[curY]
            c = B_comp[curY]
            cnt_le = BIT_A_cnt.sum(c)
            sum_le = BIT_A_sum.sum(c)
            curAns -= v*cnt_le - sum_le + (sumA - sum_le) - v*(curX - cnt_le)
            BIT_B_cnt.add(c, -1)
            BIT_B_sum.add(c, -v)
            sumB -= v
        res[qi] = curAns
    # output
    out = sys.stdout
    for v in res:
        out.write(str(v) + "
")

if __name__ == "__main__":
    main()