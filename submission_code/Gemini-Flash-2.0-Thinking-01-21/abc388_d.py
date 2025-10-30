import sys

class FenwickTree:
    def __init__(self, size):
        # Tree is 1-indexed, covering original values 0 to size-2
        self.size = size
        self.tree = [0] * size

    def add(self, value, delta):
        # Add delta to the count of `value`. value is 0-indexed.
        # Map value `v` to index `v + 1` in the 1-indexed tree.
        idx = value + 1
        while idx < self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, value):
        # Get the sum of frequencies for values from 0 up to `value` (inclusive).
        # Map value `v` to index `v + 1` in the 1-indexed tree. Query up to index `value + 1`.
        idx = value + 1
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

# Use query_le(v) for sum <= v (0-indexed value v)
def query_le(ft, v):
    if v < 0:
        return 0
    # ft.query(v) sums frequencies for original values 0 up to v
    return ft.query(v)

# Use query_lt(v) for sum < v (0-indexed value v)
def query_lt(ft, v):
    # sum < v is sum <= v-1
    return query_le(ft, v - 1)

# Use query_ge(v) for sum >= v (0-indexed value v)
def query_ge(ft, v, total_count):
    # sum >= v is total_count - sum < v
    return total_count - query_lt(ft, v)


def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Max possible value of R_i = A[i] + gifts_received[i] + i (0-indexed)
    # gifts_received[i] <= i <= N-1
    # Max R_i <= max(A[i]) + (N-1) + (N-1) approx 5e5 + 2*5e5 = 1.5e6
    # Set BIT size to accommodate values up to approx 1.5e6.
    # Use a safe bound like 2 * 10^6.
    # The maximum value that can be stored in the BIT or queried is relevant for BIT_SIZE.
    # Max R_i approx 1.5e6. Max query threshold i approx 5e5.
    # So values in BIT are within [0, ~1.5e6].
    MAX_VALUE_IN_BIT = 2 * 10**6 # Safe upper bound for values stored/queried in BIT
    BIT_SIZE = MAX_VALUE_IN_BIT + 2 # Fenwick tree size for values 0 to MAX_VALUE_IN_BIT


    ft = FenwickTree(BIT_SIZE)
    total_r_count = 0 # Keep track of total elements added to BIT
    gifts_received = [0] * N

    for i in range(N):
        # Alien i becomes adult at year i + 1.
        # Gifts received by alien i = count { j in [0, i-1] | R_j >= i }
        # R_j = A[j] + gifts_received[j] + j
        # We need count of R_j values (for j < i) that are >= i.
        # Use query_ge(threshold, total_count_so_far).
        # The threshold is `i` (0-indexed).

        # `total_r_count` is the count of R_k values inserted so far (for k in [0, i-1]).
        # Query for values >= i.
        count_ge_i = query_ge(ft, i, total_r_count)
        gifts_received[i] = count_ge_i

        # Calculate R_i = A[i] + gifts_received[i] + i and add to BIT
        R_i = A[i] + gifts_received[i] + i
        # Ensure R_i is within bounds for the BIT.
        # Since MAX_VALUE_IN_BIT is >= max possible R_i, this is fine.
        ft.add(R_i, 1)
        total_r_count += 1

    B = [0] * N
    for i in range(N):
        # Final stones B[i] = A[i] + gifts_received[i] - gifts_given_by_i
        # gifts_given_by_i = min(stones_i_at_adulthood, number_of_giving_events_after_i)
        # stones_i_at_adulthood = A[i] + gifts_received[i]
        # number_of_giving_events_after_i = N - 1 - i (aliens i+1, ..., N-1)
        # This is the number of other aliens who become adults after alien i.

        stones_at_adulthood_i = A[i] + gifts_received[i]
        giving_opportunities_after_i = N - 1 - i

        # gifts_given_by_i is the total number of stones alien i gives away.
        # This is limited by its stone count at adulthood, and the number of times it's required to give (N-1-i events).
        # It gives 1 stone per event if it has > 0 stones.
        # The total given is min(stones_at_adulthood_i, N-1-i), because once it runs out of stones, it stops giving.
        gifts_given_by_i = min(stones_at_adulthood_i, giving_opportunities_after_i)

        B[i] = stones_at_adulthood_i - gifts_given_by_i

    # Print output separated by spaces
    print(*B)

solve()