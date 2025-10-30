import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

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

    def range_add(self, a, b, delta):
        self.update(a, delta)
        if b + 1 <= self.n:
            self.update(b + 1, -delta)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    contests = []
    for _ in range(N):
        L = int(input[ptr])
        R = int(input[ptr+1])
        contests.append((L, R))
        ptr += 2
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        X = int(input[ptr])
        queries.append(X)
        ptr += 1

    max_X = 500000
    ft = FenwickTree(max_X)

    for L, R in contests:
        # Compute A_i: smallest X where S(X) >= L
        low, high = 1, max_X
        A = max_X + 1
        while low <= high:
            mid = (low + high) // 2
            s = mid + ft.query(mid)
            if s >= L:
                A = mid
                high = mid - 1
            else:
                low = mid + 1
        # Compute B_i: largest X where S(X) <= R
        low, high = 1, max_X
        B = 0
        while low <= high:
            mid = (low + high) // 2
            s = mid + ft.query(mid)
            if s <= R:
                B = mid
                low = mid + 1
            else:
                high = mid - 1
        if A <= B:
            ft.range_add(A, B, 1)

    for X in queries:
        total = ft.query(X)
        print(X + total)

if __name__ == "__main__":
    main()