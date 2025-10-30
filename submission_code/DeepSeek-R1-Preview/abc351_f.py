import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

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
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    sorted_unique = sorted(list(set(A)))
    max_rank = len(sorted_unique)
    sum_tree = FenwickTree(max_rank)
    cnt_tree = FenwickTree(max_rank)
    total = 0
    for ai in reversed(A):
        r = bisect.bisect_left(sorted_unique, ai) + 1
        sum_gt = sum_tree.query(max_rank) - sum_tree.query(r)
        cnt_gt = cnt_tree.query(max_rank) - cnt_tree.query(r)
        total += sum_gt - ai * cnt_gt
        sum_tree.update(r, ai)
        cnt_tree.update(r, 1)
    print(total)

if __name__ == '__main__':
    main()