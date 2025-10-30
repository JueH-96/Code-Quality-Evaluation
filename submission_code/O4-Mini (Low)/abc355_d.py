import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input())
    intervals = []
    coords = []
    for _ in range(N):
        l, r = map(int, input().split())
        intervals.append((l, r))
        coords.append(l)
        coords.append(r)
    # sort intervals by l ascending
    intervals.sort(key=lambda x: x[0])
    # coordinate compress all l and r
    coords = list(sorted(set(coords)))
    comp = {v: i+1 for i, v in enumerate(coords)}  # 1-based for BIT
    size = len(coords)

    # Fenwick Tree / BIT for counting r's
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n+1)
        def add(self, i, v):
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def sum(self, i):
            s = 0
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_sum(self, l, r):
            if r < l:
                return 0
            return self.sum(r) - self.sum(l-1)

    bit = Fenwick(size)
    non_intersecting = 0
    # sweep intervals in increasing l
    for l, r in intervals:
        # count previous intervals with r_i < l_j
        # find compressed position just below l
        pos_l = comp[l]
        # r < l  => compressed r index < pos_l
        cnt = bit.sum(pos_l - 1)
        non_intersecting += cnt
        # now add this interval's r to BIT
        pos_r = comp[r]
        bit.add(pos_r, 1)

    total_pairs = N * (N-1) // 2
    intersecting = total_pairs - non_intersecting
    print(intersecting)

if __name__ == "__main__":
    main()