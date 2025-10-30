import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    H = list(map(int, input().split()))

    # Compute L[j]: the nearest index to the left of j with H[L[j]] > H[j], or 0 if none
    L = [0] * (N + 1)  # 1-based
    stack = []
    for j in range(1, N + 1):
        h = H[j - 1]
        while stack and H[stack[-1] - 1] < h:
            stack.pop()
        L[j] = stack[-1] if stack else 0
        stack.append(j)

    # Prepare points (L[j], j)
    pts = [(L[j], j) for j in range(1, N + 1)]
    pts.sort(key=lambda x: x[0])

    # Read queries, store (l, r, idx)
    queries = []
    for i in range(Q):
        l, r = map(int, input().split())
        queries.append((l, r, i))
    # Sort queries by l
    queries.sort(key=lambda x: x[0])

    # Fenwick / BIT for counts on positions 1..N
    class BIT:
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n + 1)
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

    bit = BIT(N)
    ans = [0] * Q
    p = 0
    # Process each query in increasing l
    for l, r, qi in queries:
        # Add all pts with L[j] <= l
        while p < N and pts[p][0] <= l:
            _, jpos = pts[p]
            bit.add(jpos, 1)
            p += 1
        # Count j > r: it's total added minus sum up to r
        total = bit.sum(N)
        left_r = bit.sum(r)
        ans[qi] = total - left_r

    # Output in original order
    out = sys.stdout
    for v in ans:
        out.write(str(v) + "
")

if __name__ == "__main__":
    main()