import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]

    # Coordinate compression
    uniq = sorted(set(A))
    m = len(uniq)
    comp = {v: i+1 for i, v in enumerate(uniq)}  # 1-based indices for Fenwick

    # Fenwick trees: one for counts, one for sums
    class Fenwick:
        def __init__(self, size):
            self.n = size
            self.fw = [0] * (size + 1)
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

    fw_count = Fenwick(m)
    fw_sum   = Fenwick(m)

    ans = 0
    for x in A:
        idx = comp[x]
        # query number of previous < x, and their sum
        cnt = fw_count.sum(idx - 1)
        s   = fw_sum.sum(idx - 1)
        # contribute cnt * x - s
        ans += cnt * x - s
        # now include x in trees
        fw_count.add(idx, 1)
        fw_sum.add(idx, x)

    print(ans)

if __name__ == "__main__":
    main()