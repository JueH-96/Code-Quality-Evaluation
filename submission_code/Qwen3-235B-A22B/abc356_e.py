import math
import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)
    
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

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:]))
    A.sort()
    MAX_A = 10**6
    ft = FenwickTree(MAX_A)
    cnt = [0] * (MAX_A + 2)
    total = 0
    for y in A:
        sqrt_y = int(math.isqrt(y))
        sum_p1 = 0
        # Compute sum_part1
        max_x = sqrt_y
        for x in range(1, max_x + 1):
            if cnt[x]:
                sum_p1 += cnt[x] * (y // x)
        # Compute sum_part2
        sum_p2 = 0
        for k in range(1, sqrt_y + 1):
            L = (y // (k + 1)) + 1
            R = y // k
            lower_bound = max(L, sqrt_y + 1)
            if lower_bound > R:
                continue
            # Query BIT for [lower_bound, R]
            sum_p2 += k * (ft.query(R) - ft.query(lower_bound - 1))
        total += sum_p1 + sum_p2
        # Update the frequency and Fenwick Tree
        cnt[y] += 1
        ft.update(y, 1)
    print(total)

if __name__ == '__main__':
    main()