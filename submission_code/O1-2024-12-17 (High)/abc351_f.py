def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Coordinate compression
    unique_vals = sorted(set(A))
    rank = {}
    for i, v in enumerate(unique_vals):
        rank[v] = i + 1  # 1-based indexing

    # Fenwick (Binary Indexed) Tree implementation
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.data = [0] * (size + 1)

        def update(self, i, val):
            while i <= self.size:
                self.data[i] += val
                i += i & -i

        def query(self, i):
            s = 0
            while i > 0:
                s += self.data[i]
                i -= i & -i
            return s

    # Two Fenwicks: one for counts, one for sums
    fen_count = FenwickTree(len(unique_vals))
    fen_sum = FenwickTree(len(unique_vals))

    ans = 0
    for x in A:
        r = rank[x]
        # Count how many previous elements < x
        cnt_less = fen_count.query(r - 1) if r > 1 else 0
        sum_less = fen_sum.query(r - 1) if r > 1 else 0

        # Contribution for current x (A_j) from all A_i < x
        ans += x * cnt_less - sum_less

        # Update Fenwicks with current value
        fen_count.update(r, 1)
        fen_sum.update(r, x)

    print(ans)

# Do NOT forget to call main at the end
if __name__ == "__main__":
    main()