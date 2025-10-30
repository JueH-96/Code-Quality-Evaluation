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
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    intervals = []
    for _ in range(N):
        l = int(input[ptr])
        r = int(input[ptr+1])
        intervals.append((l, r))
        ptr += 2
    # Sort intervals by their starting point
    intervals.sort()
    # Collect all l and r values for coordinate compression
    all_values = []
    for l, r in intervals:
        all_values.append(l)
        all_values.append(r)
    # Create sorted_unique list
    sorted_all = sorted(all_values)
    sorted_unique = []
    prev = None
    for val in sorted_all:
        if val != prev:
            sorted_unique.append(val)
            prev = val
    # Initialize Fenwick Tree
    ft_size = len(sorted_unique)
    ft = FenwickTree(ft_size)
    total_non_overlapping = 0
    for l_j, r_j in intervals:
        x = l_j - 1
        # Compute the number of elements <= x in sorted_unique
        q_idx = bisect.bisect_right(sorted_unique, x)
        count = ft.query(q_idx)
        total_non_overlapping += count
        # Insert r_j into Fenwick Tree
        pos = bisect.bisect_left(sorted_unique, r_j) + 1  # 1-based index
        ft.update(pos, 1)
    total_pairs = N * (N - 1) // 2
    answer = total_pairs - total_non_overlapping
    print(answer)

if __name__ == "__main__":
    main()