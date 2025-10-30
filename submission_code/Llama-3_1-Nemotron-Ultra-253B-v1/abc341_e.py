import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    B = [0] * (N + 2)  # B[1..N-1] are used
    ft = FenwickTree(N-1)
    for i in range(1, N):
        if S[i-1] == S[i]:
            B[i] = 1
            ft.add(i, 1)

    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        if query_type == '1':
            # Type 1: flip L to R
            if L > 1:
                i = L - 1
                if B[i] == 1:
                    ft.add(i, -1)
                    B[i] = 0
                else:
                    ft.add(i, 1)
                    B[i] = 1
            if R < N:
                i = R
                if B[i] == 1:
                    ft.add(i, -1)
                    B[i] = 0
                else:
                    ft.add(i, 1)
                    B[i] = 1
        else:
            # Type 2: query L to R
            if L == R:
                print("Yes")
            else:
                sum_val = ft.query(R-1) - ft.query(L-1)
                print("Yes" if sum_val == 0 else "No")

if __name__ == "__main__":
    main()