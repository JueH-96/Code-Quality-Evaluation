import math

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    A.sort()
    max_x = 10**6
    ft = FenwickTree(max_x)
    total_sum = 0
    for x in A:
        K = int(math.isqrt(x))
        current_sum = 0
        # Process q from 1 to K, for d > K
        for q in range(1, K + 1):
            lower = x / (q + 1)
            upper = x / q
            d_start = math.floor(lower) + 1
            d_end = math.floor(upper)
            d_start = max(d_start, K + 1)
            if d_start > d_end:
                continue
            count = ft.range_query(d_start, d_end)
            current_sum += q * count
        # Process d from 1 to K
        for d in range(1, K + 1):
            count = ft.range_query(d, d)
            current_sum += (x // d) * count
        total_sum += current_sum
        ft.update(x, 1)
    print(total_sum)

if __name__ == '__main__':
    main()