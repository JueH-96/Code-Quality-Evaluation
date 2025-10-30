import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def add(self, idx, val):
        # idx is 0-based
        idx += 1
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        # sum from 0..idx (0-based)
        idx += 1
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_add(self, l, r, val):
        self.add(l, val)
        if r + 1 < self.n:
            self.add(r + 1, -val)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N
    B_list = list(map(int, input[ptr:ptr + M]))
    
    sum_take = [0] * N
    total_q = 0
    ft = FenwickTree(N)
    
    for B in B_list:
        current = A[B] - sum_take[B] + total_q + ft.query(B)
        K = current
        sum_take[B] += K
        q, r = divmod(K, N)
        total_q += q
        if r != 0:
            S = (B + 1) % N
            start = S
            if start + r <= N:
                a = start
                b = start + r - 1
                ft.range_add(a, b, 1)
            else:
                a1 = start
                b1 = N - 1
                ft.range_add(a1, b1, 1)
                a2 = 0
                b2 = (start + r - 1) - N
                ft.range_add(a2, b2, 1)
    
    res = []
    for j in range(N):
        val = A[j] - sum_take[j] + total_q + ft.query(j)
        res.append(str(val))
    print(' '.join(res))

if __name__ == '__main__':
    main()