import sys

# Fenwick Tree (Binary Indexed Tree) implementation
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1) # BIT is 1-indexed, so size + 1

    def update(self, index, delta):
        # index is 0-based, convert to 1-based for BIT
        index += 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index):
        # index is 0-based, convert to 1-based for BIT
        # query sum from 0 to index (inclusive)
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    total_ans = 0

    # Fenwick trees for counts and sums of prefix sums modulo M
    # size M, to store frequencies/sums for values from 0 to M-1
    bit_count = FenwickTree(M)
    bit_sum = FenwickTree(M)

    # P_0 = 0. Its modulo M value is 0.
    # We include P_0 = 0 in our initial set of prefix sums for previous k values.
    bit_count.update(0, 1) # Count of value 0 is 1
    bit_sum.update(0, 0)   # Sum of value 0 is 0

    current_prefix_sum = 0 # This stores P_k (the actual prefix sum, not modulo M)

    # The loop variable 'i' here corresponds to the (current_r_index - 1)
    # The current prefix sum P_{i+1} is derived from A[i].
    for i in range(N):
        current_prefix_sum += A[i]
        P_prime_current_r = current_prefix_sum % M

        # Calculate contributions for current P_prime_current_r
        # We need to sum (P_prime_current_r - P_prime_k + M) % M for all k in {0, ..., i}
        # (These P_prime_k values are already in the BITs at this point)

        # Case 1: P_prime_k <= P_prime_current_r
        # Term is (P_prime_current_r - P_prime_k)
        # Sum = (count of P_prime_k <= P_prime_current_r) * P_prime_current_r - (sum of such P_prime_k)
        num_le = bit_count.query(P_prime_current_r)
        sum_le_P_prime_k = bit_sum.query(P_prime_current_r)

        total_ans += (num_le * P_prime_current_r - sum_le_P_prime_k)

        # Case 2: P_prime_k > P_prime_current_r
        # Term is (P_prime_current_r - P_prime_k + M)
        # Sum = (count of P_prime_k > P_prime_current_r) * (P_prime_current_r + M) - (sum of such P_prime_k)
        
        # Total count of P_prime_k values seen so far (from P_0 to P_i)
        total_count_seen = bit_count.query(M - 1) 
        # Total sum of P_prime_k values seen so far
        total_sum_seen = bit_sum.query(M - 1)

        num_gt = total_count_seen - num_le
        sum_gt_P_prime_k = total_sum_seen - sum_le_P_prime_k

        total_ans += (num_gt * (P_prime_current_r + M) - sum_gt_P_prime_k)

        # Add current P_prime_current_r to the Fenwick trees for future iterations (for A[i+1] onwards)
        bit_count.update(P_prime_current_r, 1)
        bit_sum.update(P_prime_current_r, P_prime_current_r)

    print(total_ans)

# Read input and solve
solve()