class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)

    def update(self, idx, delta):
        idx += 1
        while idx <= self.size + 1:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_query(self, a, b):
        if a > b:
            return 0
        return self.query(b) - self.query(a - 1)

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

    fenwick = FenwickTree(M)
    fenwick.update(0, 1)
    sum_prev = 0
    total_prev = 1
    current_prefix = 0
    total_sum = 0

    for a in A:
        current_prefix = (current_prefix + a) % M
        x = current_prefix
        count_gt_x = fenwick.range_query(x + 1, M - 1)
        contribution = x * total_prev - sum_prev + M * count_gt_x
        total_sum += contribution
        fenwick.update(x, 1)
        sum_prev += x
        total_prev += 1

    print(total_sum)

if __name__ == '__main__':
    main()