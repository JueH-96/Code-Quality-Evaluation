import sys
from bisect import bisect_left

MAXV = 1 << 20

class SegTree:
    def __init__(self, N, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.size = 2 ** (N-1).bit_length() + 1
        self.data = [alpha] * self.size

    def update(self, pos, value):
        pos += self.size - 1
        self.data[pos] = max(value, self.data[pos])
        pos //= 2
        while pos:
            if self.beta:
                self.data[pos] = max(self.data[pos*2], self.data[pos*2+1] + self.beta)
            else:
                self.data[pos] = max(self.data[pos*2], self.data[pos*2+1] * self.alpha)
            pos //= 2

    def find(self, start, end):
        return self._find(start, end, 0, self.size // 2, 1)

    def _find(self, start, end, l, r, pos):
        if start <= l and r <= end:
            return self.data[pos]

        mid = (l + r) // 2
        if end <= mid:
            return self._find(start, end, l, mid, pos * 2)
        if mid < start:
            return self._find(start, end, mid, r, pos * 2 + 1)
        return self.alpha * max(self._find(start, end, l, mid, pos * 2), self._find(start, end, mid, r, pos * 2 + 1)) + self.beta

def solve_case():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]
    
    seg_tree = SegTree(N + 2, 1, 0)
    seg_tree.update(0, 0)
    for i in range(N):
        seg_tree.update(i + 1, max(seg_tree.find(i, i + 1) * B[i], seg_tree.find(i, i + 1) + A[i]))
    
    for q in queries:
        if q[0] == 1:
            i, x = q[1] - 1, q[2]
            A[i] = x
        elif q[0] == 2:
            i, x = q[1] - 1, q[2]
            B[i] = x
        else:
            l, r = q[1] - 1, q[2]
            maxv = 0
            for i in range(l, r + 1):
                maxv = max(seg_tree.find(l, i) * B[i], seg_tree.find(l, i) + A[i])
                seg_tree.update(i + 1, maxv)
            seg_tree.update(l, 0)
            print(maxv)

solve_case()