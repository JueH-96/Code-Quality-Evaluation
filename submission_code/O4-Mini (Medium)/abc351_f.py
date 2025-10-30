import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    # Coordinate compression
    xs = sorted(set(A))
    m = len(xs)
    comp = {x: i+1 for i, x in enumerate(xs)}  # 1-based indices for BIT

    # Fenwick / BIT for counts and sums
    class BIT:
        __slots__ = ('n','tree')
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n+1)
        def add(self, i, v):
            # add v at index i
            n = self.n
            tr = self.tree
            while i <= n:
                tr[i] += v
                i += i & -i
        def sum(self, i):
            # sum of [1..i]
            s = 0
            tr = self.tree
            while i > 0:
                s += tr[i]
                i -= i & -i
            return s

    bit_count = BIT(m)
    bit_sum = BIT(m)
    ans = 0

    # Process each A[j]
    for x in A:
        idx = comp[x]
        # count of previous values < x
        cnt = bit_count.sum(idx - 1)
        s = bit_sum.sum(idx - 1)
        # contribution
        ans += cnt * x - s
        # add current
        bit_count.add(idx, 1)
        bit_sum.add(idx, x)

    # output
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()