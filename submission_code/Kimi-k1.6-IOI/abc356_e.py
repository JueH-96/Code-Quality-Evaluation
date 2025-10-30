import math

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing
    
    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    a.sort()
    max_a = 10**6
    fenwick = FenwickTree(max_a)
    ans = 0
    for y in a:
        m = int(math.isqrt(y))
        sum_part1 = 0
        for x in range(1, m + 1):
            cnt = fenwick.query(x)
            sum_part1 += (y // x) * cnt
        sum_part2 = 0
        for k in range(1, m + 1):
            a_start = (y // (k + 1)) + 1
            a_start = max(a_start, m + 1)
            a_end = y // k
            if a_start > a_end:
                continue
            cnt = fenwick.query(a_end) - fenwick.query(a_start - 1)
            sum_part2 += k * cnt
        total = sum_part1 + sum_part2
        ans += total
        fenwick.update(y, 1)
    print(ans)

if __name__ == "__main__":
    main()