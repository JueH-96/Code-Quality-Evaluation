import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update_point(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query_prefix(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_query(self, l, r):
        return self.query_prefix(r) - self.query_prefix(l - 1)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    S = ' ' + S  # 1-based indexing

    if N == 1:
        for _ in range(Q):
            t = int(input[ptr])
            ptr += 1
            L = int(input[ptr])
            ptr += 1
            R = int(input[ptr])
            ptr += 1
            if t == 2:
                print("Yes")
        return

    n_diff = N - 1
    current = [0] * (n_diff + 2)  # 1-based to n_diff
    ft = FenwickTree(n_diff)

    for i in range(1, n_diff + 1):
        if S[i] != S[i + 1]:
            current[i] = 1
        else:
            current[i] = 0
        ft.update_point(i, current[i])

    for _ in range(Q):
        t = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if t == 1:
            if L > 1:
                pos = L - 1
                val = current[pos]
                delta = 1 - 2 * val
                ft.update_point(pos, delta)
                current[pos] = 1 - val
            if R < N:
                pos = R
                val = current[pos]
                delta = 1 - 2 * val
                ft.update_point(pos, delta)
                current[pos] = 1 - val
        else:
            if L == R:
                print("Yes")
            else:
                a = L
                b = R - 1
                sum_val = ft.range_query(a, b)
                if sum_val == (b - a + 1):
                    print("Yes")
                else:
                    print("No")

if __name__ == '__main__':
    main()