import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Precompute cnt and sum_pos arrays
    cnt = [0] * M
    sum_pos = [0] * M
    for i in range(N):
        val = A[i]
        cnt[val] += 1
        sum_pos[val] += (i + 1)  # positions are 1-based

    # Compute initial inversion count (inv0) using Fenwick Tree
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)  # 1-based indexing

        def update_point(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx

        def query_prefix(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    ft = FenwickTree(M)
    inv0 = 0
    current_count = 0
    for a in A:
        q = ft.query_prefix(a + 1)
        inv0 += (current_count - q)
        ft.update_point(a + 1, 1)
        current_count += 1

    answers = [0] * M
    answers[0] = inv0

    # Compute answers for k=1 to M-1
    for k in range(M - 1):
        v = (M - 1) - k
        total_m1 = cnt[v]
        sum_pos_v = sum_pos[v]
        delta = 2 * sum_pos_v - total_m1 * (N + 1)
        answers[k + 1] = answers[k] + delta

    for ans in answers:
        print(ans)

if __name__ == "__main__":
    main()