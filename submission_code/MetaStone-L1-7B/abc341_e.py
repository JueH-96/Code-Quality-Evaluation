import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, stdin.readline().split())
    S = list(map(int, stdin.readline().split()))
    S = [0] + S  # 1-based

    # Compute initial A array
    A = [0] * (N + 1)  # 1-based
    for i in range(1, N):
        A[i] = 1 if S[i] != S[i+1] else 0

    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.sum = [0] * (2 * self.size)
            self.lazy = [0] * (2 * self.size)
            for i in range(self.n):
                self.sum[self.size + i] = data[i]
            for i in range(self.size - 1, 0, -1):
                self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]

        def push(self, node, l, r):
            if self.lazy[node] != 0:
                self.sum[node] ^= self.lazy[node]
                if l != r:
                    self.lazy[2 * node] ^= self.lazy[node]
                    self.lazy[2 * node + 1] ^= self.lazy[node]
                self.lazy[node] = 0

        def update_range(self, node, l, r, ul, ur, val):
            self.push(node, l, r)
            if ur < l or ul > r:
                return
            if ul <= l and r <= ur:
                self.sum[node] ^= (r - l + 1)
                self.lazy[node] ^= 1
                return
            mid = (l + r) // 2
            self.update_range(2 * node, l, mid, ul, ur, val)
            self.update_range(2 * node + 1, mid + 1, r, ul, ur, val)
            self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

        def query_range(self, node, l, r, ql, qr):
            self.push(node, l, r)
            if qr < l or ql > r:
                return 0
            if ql <= l and r <= qr:
                return self.sum[node]
            mid = (l + r) // 2
            return self.query_range(2 * node, l, mid, ql, qr) + self.query_range(2 * node + 1, mid + 1, r, ql, qr)

    # Initialize segment tree
    data = [0] * (N + 2)
    for i in range(1, N+1):
        data[i] = A[i]
    st = SegmentTree(data)

    for _ in range(Q):
        query = stdin.readline().split()
        if not query:
            continue
        if query[0] == '1':
            L = int(query[1])
            R = int(query[2])
            L0 = L
            R0 = R
            if L0 <= R0:
                st.update_range(1, 1, st.size, L0, R0, 1)
        else:
            L = int(query[1])
            R = int(query[2])
            L0 = L
            R0 = R
            if L0 > R0:
                print("Yes")
                continue
            if L0 == R0:
                print("Yes")
                continue
            sum_A = st.query_range(1, 1, st.size, L0, R0-1)
            if sum_A == (R0 - L0):
                print("Yes")
            else:
                print("No")

if __name__ == '__main__':
    main()