import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, x, delta):
        while x <= self.n:
            self.tree[x] += delta
            x += x & -x

    def query(self, x):
        res = 0
        while x > 0:
            res += self.tree[x]
            x -= x & -x
        return res

    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    intervals = []
    for _ in range(N):
        A = int(input[ptr])
        B = int(input[ptr+1])
        ptr += 2
        u = min(A, B)
        v = max(A, B)
        intervals.append((u, v))
    
    intervals.sort()
    max_val = 2 * N
    ft = FenwickTree(max_val)
    
    for a, b in intervals:
        L = a + 1
        R = b - 1
        if L <= R:
            count = ft.range_query(L, R)
            if count > 0:
                print("Yes")
                return
        ft.update(b, 1)
    
    print("No")

if __name__ == "__main__":
    main()