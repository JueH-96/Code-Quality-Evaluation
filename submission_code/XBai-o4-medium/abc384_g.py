import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based indexing

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
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N
    B = list(map(int, input[ptr:ptr + N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    queries = []
    for i in range(K):
        X = int(input[ptr])
        Y = int(input[ptr + 1])
        queries.append((X, Y, i))
        ptr += 2

    # Prepare all_values for coordinate compression
    all_values = A + B
    unique_sorted = sorted(list(set(all_values)))
    max_rank = len(unique_sorted)

    # Precompute prefix_B
    prefix_B = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_B[i] = prefix_B[i - 1] + B[i - 1]

    # Precompute pre_a_bisect
    pre_a_bisect = [0] * N
    for i in range(N):
        pre_a_bisect[i] = bisect.bisect_right(unique_sorted, A[i])

    # Precompute pre_b_bisect
    pre_b_bisect = [0] * N
    for i in range(N):
        pre_b_bisect[i] = bisect.bisect_left(unique_sorted, B[i])

    # Sort queries by Y, and keep track of original index
    sorted_queries = sorted(queries, key=lambda x: x[1])

    # Prepare to process queries offline
    # Initialize Fenwick Trees
    count_bit = FenwickTree(max_rank)
    sum_bit = FenwickTree(max_rank)

    # Current Y processed
    current_y = 0

    # Answers array
    answers = [0] * K

    # Process queries grouped by Y
    i = 0
    while i < K:
        current_group_Y = sorted_queries[i][1]
        # Update current_y to current_group_Y
        while current_y < current_group_Y:
            current_y += 1
            b_val = B[current_y - 1]
            # Find rank for b_val
            pos = pre_b_bisect[current_y - 1]
            rank = pos + 1  # 1-based
            # Update Fenwick Trees
            count_bit.update(rank, 1)
            sum_bit.update(rank, b_val)
        # Compute prefix_sums for this Y
        prefix_sums = [0] * (N + 1)  # 1-based
        for x in range(1, N + 1):
            a_val = A[x - 1]
            r = pre_a_bisect[x - 1]
            count_le = count_bit.query(r)
            sum_le = sum_bit.query(r)
            sum_total = prefix_B[current_group_Y]
            current = a_val * (2 * count_le - current_group_Y) + (sum_total - 2 * sum_le)
            prefix_sums[x] = prefix_sums[x - 1] + current
        # Process all queries with this Y
        j = i
        while j < K and sorted_queries[j][1] == current_group_Y:
            X, Y, idx = sorted_queries[j]
            answers[idx] = prefix_sums[X]
            j += 1
        i = j

    # Output answers in original order
    for ans in answers:
        print(ans)

if __name__ == "__main__":
    main()