import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # Using 1-based indexing

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

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = list(input[ptr])
    ptr += 1

    T_val = []
    ft = None

    if N >= 2:
        T_val = [0] * (N - 1)
        for i in range(N - 1):
            if S[i] != S[i + 1]:
                T_val[i] = 1
            else:
                T_val[i] = 0
        ft = FenwickTree(N - 1)
        for i in range(N - 1):
            ft.update(i + 1, T_val[i])

    output = []

    for _ in range(Q):
        type_q = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if type_q == 1:
            if N == 1:
                continue
            A = L - 1
            B = R - 1
            # Check and update T[A-1]
            if A > 0:
                pos = A - 1
                old = T_val[pos]
                new_val = 1 - old
                delta = new_val - old
                ft.update(pos + 1, delta)
                T_val[pos] = new_val
            # Check and update T[B]
            if B < N - 1:
                pos = B
                old = T_val[pos]
                new_val = 1 - old
                delta = new_val - old
                ft.update(pos + 1, delta)
                T_val[pos] = new_val
        else:
            if L == R:
                output.append("Yes")
            else:
                a = L - 1
                b = R - 2
                sum_val = ft.query(b + 1) - ft.query(a)
                if sum_val == (b - a + 1):
                    output.append("Yes")
                else:
                    output.append("No")

    print('
'.join(output))

if __name__ == "__main__":
    main()