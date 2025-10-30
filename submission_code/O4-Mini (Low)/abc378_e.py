import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # Fenwick / BIT for counts and sums over [0..M-1], 1-indexed of size M
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n+1)
        def add(self, i, v):
            # i: 1-based
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
            if r < l:
                return 0
            return self.sum(r) - self.sum(l-1)

    # Two trees: one for counts, one for sums of values
    bit_cnt = Fenwick(M)
    bit_sum = Fenwick(M)

    # Initialize with prefix sum S_0 = 0
    # we store at index 1 (value 0 maps to idx=1)
    bit_cnt.add(1, 1)
    bit_sum.add(1, 0)
    total_count = 1
    total_sum = 0

    ans = 0
    prefix = 0

    for a in A:
        prefix = (prefix + a) % M
        # map prefix value x in [0..M-1] to idx = x+1
        x = prefix
        idx = x + 1

        # count of previous y <= x:
        c_le = bit_cnt.sum(idx)
        s_le = bit_sum.sum(idx)
        # remaining are y > x
        c_gt = total_count - c_le
        s_gt = total_sum - s_le

        # sum of (x - y) for y<=x plus (x - y + M) for y>x
        # = c_le*x - s_le + c_gt*(x+M) - s_gt
        contrib = c_le * x - s_le + c_gt * (x + M) - s_gt
        ans += contrib

        # now insert this prefix into BITs
        bit_cnt.add(idx, 1)
        bit_sum.add(idx, x)
        total_count += 1
        total_sum += x

    print(ans)

if __name__ == "__main__":
    main()