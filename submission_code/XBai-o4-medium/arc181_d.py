import sys
import threading
from bisect import bisect_left, bisect_right
import math
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute for range maximum query with position
    # Sparse table for range maximum query
    logn = 17
    st = [[(0, 0)]*(N+1) for _ in range(logn+1)]
    for i in range(1, N+1):
        st[0][i] = (P[i-1], i)
    for k in range(1, logn+1):
        for i in range(1, N+1 - (1 << k) + 1):
            mid = i + (1 << (k-1))
            left = st[k-1][i]
            right = st[k-1][mid]
            if left[0] > right[0]:
                st[k][i] = left
            else:
                st[k][i] = right

    def get_max(l, r):
        # l and r are 1-based
        length = r - l + 1
        k = length.bit_length() - 1
        mid = r - (1 << k) + 1
        a = st[k][l]
        b = st[k][mid]
        if a[0] > b[0]:
            return a
        else:
            return b

    # Initial inversion count using Fenwick Tree
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0]*(self.n + 2)
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
        def range_query(self, l, r):
            return self.query(r) - self.query(l-1)

    max_val = max(P)
    ft = FenwickTree(max_val + 2)
    inv_count = 0
    for i in range(N-1, -1, -1):
        inv_count += ft.query(P[i]-1)
        ft.update(P[i], 1)

    current_P = [0] * (N + 1)  # 1-based
    last = 0

    for a in A:
        k = a
        if last == 0:
            a_range = 1
        else:
            a_range = last + 1
        b_range = k
        orig_max_val, orig_max_pos = get_max(a_range, b_range)
        if last == 0:
            current_max_val = orig_max_val
            current_max_pos = orig_max_pos
        else:
            current_P_last = current_P[last]
            if current_P_last > orig_max_val:
                current_max_val = current_P_last
                current_max_pos = last
            else:
                current_max_val = orig_max_val
                current_max_pos = orig_max_pos
        S = k - current_max_pos
        inv_count -= S
        print(inv_count)
        current_P[k] = current_max_val
        last = k

threading.Thread(target=main).start()