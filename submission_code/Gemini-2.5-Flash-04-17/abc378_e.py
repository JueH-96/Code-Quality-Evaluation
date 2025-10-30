import sys

# Fenwick tree (Binary Indexed Tree) class
# Supports 0-based values in range [0, max_val - 1]
class FenwickTree:
    def __init__(self, max_val):
        """max_val is the exclusive upper bound for values (values are 0 to max_val-1).
           Tree size is max_val. Indices 1..max_val.
        """
        self.size = max_val
        self.tree = [0] * (self.size + 1)

    def add(self, val, count):
        """Adds count to the value val (0-based)."""
        # Map 0-based value `val` to 1-based index `val + 1`
        idx = val + 1
        while idx <= self.size:
            self.tree[idx] += count
            idx += idx & (-idx)

    def query(self, val):
        """Queries the sum of counts for values from 0 up to val (0-based)."""
        # Map 0-based value `val` to 1-based index `val + 1`
        idx = val + 1
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx)
        return s

    def query_range_greater_than(self, val):
        """Queries the sum of counts for values strictly greater than val (0-based).
           Values are in range [val+1, self.size-1].
           Equivalent to total count minus count of values <= val.
        """
        # Total count is sum of counts for all values in [0, self.size - 1]
        total_count = self.query(self.size - 1)
        # Count of values <= val
        count_le_val = self.query(val)
        # Count of values > val
        return total_count - count_le_val

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # 1. Compute prefix sums modulo M
    # Q[k] will store P_k % M, where P_k = sum_{i=1..k} A_i (1-based index)
    # P_0 = 0
    Q = [0] * (N + 1)
    current_prefix_sum_mod_M = 0
    for i in range(N):
        # A is 0-indexed in code, A[i] corresponds to problem's A_{i+1}
        current_prefix_sum_mod_M = (current_prefix_sum_mod_M + A[i]) % M
        Q[i+1] = current_prefix_sum_mod_M

    # Q is now [P_0%M, P_1%M, ..., P_N%M]
    # Sum is over 1 <= l <= r <= N, of (P_r - P_{l-1}) % M
    # Let j = l-1. Range of (j, r) is 0 <= j <= r-1 and 1 <= r <= N, i.e., 0 <= j < r <= N.
    # Sum = sum_{0 <= j < r <= N} ((Q_r - Q_j + M) mod M)
    # Sum = sum_{0 <= j < r <= N} (Q_r - Q_j) + M * (number of pairs (j, r) with 0 <= j < r <= N and Q_j > Q_r)

    # 2. Calculate the first part: sum_{0 <= j < r <= N} (Q_r - Q_j)
    # This is sum_{r=1..N} r * Q_r - sum_{j=0..N-1} (N-j) * Q_j
    sum_part1 = 0
    for r in range(1, N + 1):
        sum_part1 += r * Q[r]

    for j in range(N): # j from 0 to N-1
        sum_part1 -= (N - j) * Q[j]

    # 3. Calculate the number of inversions in Q_0, ..., Q_N for pairs (j, r) with j < r
    # Number of inversions = sum_{i=0..N} (count of j < i such that Q_j > Q_i)
    inversion_count = 0
    ft = FenwickTree(M) # Values Q_i are in [0, M-1]

    for i in range(N + 1): # Iterate through Q_0, Q_1, ..., Q_N (indices 0 to N)
        # Count j < i such that Q_j > Q_i. These are values among Q_0..Q_{i-1} that are > Q_i.
        inversion_count += ft.query_range_greater_than(Q[i])

        # Add Q[i] to the Fenwick tree
        ft.add(Q[i], 1)

    # 4. Calculate the second part: M * inversion_count
    sum_part2 = M * inversion_count

    # Total answer
    answer = sum_part1 + sum_part2
    print(answer)

solve()