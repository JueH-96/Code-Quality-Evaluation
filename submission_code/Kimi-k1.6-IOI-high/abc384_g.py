import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def add(self, idx, delta):
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
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(K):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        queries.append((Y, X, _))
        ptr += 2

    # Collect all values for coordinate compression
    all_vals = A + B
    sorted_vals = sorted(list(set(all_vals)))
    M = len(sorted_vals)

    # Precompute prefix sums
    sum_A = [0] * (N + 1)
    for i in range(1, N + 1):
        sum_A[i] = sum_A[i-1] + A[i-1]
    sum_B = [0] * (N + 1)
    for i in range(1, N + 1):
        sum_B[i] = sum_B[i-1] + B[i-1]

    # Sort queries by Y then X
    queries.sort(key=lambda q: (q[0], q[1]))

    # Initialize Fenwick Trees for B's count and sum
    count_B = FenwickTree(M)
    sum_B_val = FenwickTree(M)

    # Initialize Fenwick Trees for A's count and sum
    count_A = FenwickTree(M)
    sum_A_val = FenwickTree(M)

    answers = [0] * K
    current_B = 0
    current_A = 0
    sum_AC = 0
    sum_BC = 0

    for Y, X, q_idx in queries:
        # Process B elements up to Y
        while current_B < Y:
            current_B += 1
            b_val = B[current_B - 1]
            # Compress value (1-based)
            pos = bisect.bisect_left(sorted_vals, b_val) + 1
            sum_B_val.add(pos, b_val)
            count_B.add(pos, 1)

            # Find sum of A elements >= b_val
            pos_b = bisect.bisect_left(sorted_vals, b_val)
            sum_A_ge = sum_A_val.query(M) - sum_A_val.query(pos_b)
            sum_AC += sum_A_ge

            # Count A elements < b_val
            cnt_A_lt = count_A.query(pos_b)
            sum_BC += cnt_A_lt * b_val

        # Process A elements up to X
        while current_A < X:
            current_A += 1
            a_val = A[current_A - 1]
            pos = bisect.bisect_left(sorted_vals, a_val) + 1
            sum_A_val.add(pos, a_val)
            count_A.add(pos, 1)

            # Count B elements <= a_val
            pos_b = bisect.bisect_right(sorted_vals, a_val)
            cnt_B_le = count_B.query(pos_b)
            sum_AC += cnt_B_le * a_val

            # Sum B elements > a_val
            sum_B_gt = sum_B_val.query(M) - sum_B_val.query(pos_b)
            sum_BC += sum_B_gt

        # Calculate the answer using the corrected formula
        ans = 2 * (sum_AC + sum_BC) - sum_A[X] * Y - sum_B[Y] * X
        answers[q_idx] = ans

    # Print answers in the original order
    print('
'.join(map(str, answers)))

if __name__ == "__main__":
    main()