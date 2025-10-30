class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, delta):
        idx += 1  # convert to 1-based
        while idx <= self.n + 1:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        idx += 1  # convert to 1-based
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_add(self, l, r):
        if l > r:
            return
        self.add(l, 1)
        if r + 1 < self.n:
            self.add(r + 1, -1)

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    sum_K = [0] * N
    sum_full = 0
    ft = FenwickTree(N)

    for b in B:
        i = b
        current_i = A[i] - sum_K[i] + sum_full + ft.query(i)
        K_j = current_i
        sum_K[i] += K_j
        full_j = K_j // N
        sum_full += full_j
        R_j = K_j % N

        if R_j > 0:
            S_j = (i + 1) % N
            E_j = (S_j + R_j - 1) % N
            if S_j <= E_j:
                ft.range_add(S_j, E_j)
            else:
                ft.range_add(S_j, N-1)
                ft.range_add(0, E_j)

    result = []
    for i in range(N):
        x = A[i] - sum_K[i] + sum_full + ft.query(i)
        result.append(str(x))
    print(' '.join(result))

if __name__ == '__main__':
    main()