import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    H = list(map(int, input().split()))
    # 1-based
    H = [0] + H

    # Compute L[j]: previous index k<j with H[k] > H[j], or 0 if none
    L = [0] * (N + 1)
    stack = []
    for j in range(1, N + 1):
        # pop until we find a greater height
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        L[j] = stack[-1] if stack else 0
        stack.append(j)

    # Read queries, group by r
    Qs = [[] for _ in range(N + 1)]
    ans = [0] * Q
    for qi in range(Q):
        l, r = map(int, input().split())
        # query (l, r), store in Qs[r]
        Qs[r].append((l, qi))

    # Fenwick / BIT over indices 1..N+1 for L+1
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n + 1)
        def add(self, i, v):
            # i in [1..n]
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def sum(self, i):
            # sum 1..i
            s = 0
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s

    # We need to count points j in (r..N] with L[j] <= l
    # We'll process r from N down to 1, and maintain in BIT all j>r
    # BIT indexed by L[j]+1
    bit = Fenwick(N + 1)

    current_j = N
    # iterate r = N, N-1, ..., 1
    for r in range(N, 0, -1):
        # add all j > r into BIT
        while current_j > r:
            lj = L[current_j]
            # update at index lj+1
            bit.add(lj + 1, 1)
            current_j -= 1
        # answer queries with this r
        for (l, qi) in Qs[r]:
            # count j>r with L[j] <= l  <=> L[j]+1 <= l+1
            ans[qi] = bit.sum(l + 1)

    # output answers
    w = sys.stdout.write
    for v in ans:
        w(str(v) + "
")


if __name__ == "__main__":
    main()