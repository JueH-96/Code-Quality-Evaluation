import bisect

class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # Using 1-based indexing

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

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    m = 10**8
    r = [a % m for a in A]
    unique_r = sorted(list(set(r)))
    rank_dict = {v: i+1 for i, v in enumerate(unique_r)}
    size = len(unique_r)
    bit = BIT(size)
    sum_A = sum(A)
    count_pairs = 0
    for j in range(N):
        x = m - r[j]
        if x > m - 1:
            continue
        idx = bisect.bisect_left(unique_r, x)
        if idx == len(unique_r):
            continue
        r_min = unique_r[idx]
        rk_min = rank_dict[r_min]
        total = bit.query(size)
        if rk_min > 1:
            sum_part = bit.query(rk_min - 1)
        else:
            sum_part = 0
        sum_ = total - sum_part
        count_pairs += sum_
        rj = r[j]
        rj_rank = rank_dict[rj]
        bit.update(rj_rank, 1)
    S = sum_A * (N-1) - m * count_pairs
    print(S)

if __name__ == "__main__":
    main()