import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline

    N = int(input())
    P = list(map(int, input().split()))
    M = int(input())
    A = list(map(int, input().split()))

    # Fenwick tree (BIT) for counting initial inversions
    class Fenwick:
        def __init__(self,n):
            self.n = n
            self.fw = [0]*(n+1)
        def add(self,i,v):
            while i<=self.n:
                self.fw[i]+=v
                i+=i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.fw[i]
                i-=i&-i
            return s

    fw = Fenwick(N)
    inv = 0
    # count how many > P[i] seen so far
    for i in range(N-1,-1,-1):
        x = P[i]
        inv += fw.sum(x-1)
        fw.add(x,1)

    # Maintain set of all adjacent-inversion indices i: P[i]>P[i+1]
    import bisect
    class OrderedSet:
        def __init__(self):
            self.a = []
        def add(self,x):
            i = bisect.bisect_left(self.a, x)
            if i==len(self.a) or self.a[i]!=x:
                self.a.insert(i, x)
        def discard(self,x):
            i = bisect.bisect_left(self.a, x)
            if i<len(self.a) and self.a[i]==x:
                self.a.pop(i)
        # find the smallest element < k
        def first_below(self, k):
            i = bisect.bisect_left(self.a, k)
            if i==0:
                return None
            return self.a[i-1]

    S = OrderedSet()
    for i in range(N-1):
        if P[i] > P[i+1]:
            S.add(i)

    out = []
    for k in A:
        # bubble‐pass on prefix 0..k-1:
        while True:
            i = S.first_below(k-1)
            if i is None:
                break
            # swap P[i],P[i+1]
            S.discard(i)
            P[i],P[i+1] = P[i+1],P[i]
            inv -= 1
            # fix up positions i-1, i, i+1
            for j in (i-1, i, i+1):
                if 0 <= j < N-1:
                    if P[j] > P[j+1]:
                        S.add(j)
                    else:
                        S.discard(j)
        out.append(str(inv))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()