import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query_prefix(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def query_range(self, a, b):
        if a > b:
            return 0
        return self.query_prefix(b) - self.query_prefix(a-1)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    current_D = [0] * (N)  # current_D[1..N-1] are used
    fenwick = None

    if N >= 2:
        fenwick = FenwickTree(N - 1)
        for i in range(1, N):
            if S[i-1] != S[i]:
                val = 1
            else:
                val = 0
            current_D[i] = val
            fenwick.update(i, val)

    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if query_type == '1':
            if N == 1:
                continue
            # Toggle D[L-1] if possible
            if L > 1:
                pos = L - 1
                if pos <= N - 1:
                    current_val = current_D[pos]
                    new_val = 1 - current_val
                    delta = new_val - current_val
                    fenwick.update(pos, delta)
                    current_D[pos] = new_val
            # Toggle D[R] if possible
            if R < N:
                pos = R
                if pos <= N - 1:
                    current_val = current_D[pos]
                    new_val = 1 - current_val
                    delta = new_val - current_val
                    fenwick.update(pos, delta)
                    current_D[pos] = new_val
        else:
            if L == R:
                print("Yes")
            else:
                a = L
                b = R - 1
                if N == 1:
                    print("Yes")
                else:
                    total = fenwick.query_range(a, b)
                    length = b - a + 1
                    print("Yes" if total == length else "No")

if __name__ == "__main__":
    main()