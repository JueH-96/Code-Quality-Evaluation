import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    class Fenwick:
        # 1-based Fenwick for point updates and prefix sums
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n+1)
        def add(self, i, v):
            # add v at index i
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def sum(self, i):
            # sum of [1..i]
            s = 0
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_sum(self, l, r):
            return self.sum(r) - self.sum(l-1)

    T = int(input())
    out = []
    for _ in range(T):
        n = int(input())
        A = list(map(int, input().split()))
        # Record for each value the list of positions where it appears
        pos = {}
        order = []
        for i, a in enumerate(A, start=1):
            if a not in pos:
                pos[a] = []
                order.append(a)
            pos[a].append(i)
        # Fenwick to track how many elements remain at each original position
        fw = Fenwick(n)
        for i in range(1, n+1):
            fw.add(i, 1)

        swaps = 0
        # Process values in first‐appearance order
        for x in order:
            for p in pos[x]:
                # number of other elements before p still present:
                # that's fw.sum(p-1)
                swaps += fw.sum(p-1)
            # now remove all occurrences of x
            for p in pos[x]:
                fw.add(p, -1)

        deletes = len(order)
        out.append(str(swaps + deletes))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()