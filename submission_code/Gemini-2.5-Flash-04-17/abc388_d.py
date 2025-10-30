import sys

# Use fast I/O
input = sys.stdin.readline

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        """Add delta to the element at index. Index is 1-based."""
        # Ensure index is within bounds [1, size]
        # The values added to FT are non-negative, so index will be >= 1
        if index < 1 or index > self.size:
             return # This should ideally not happen with correct MAX_POSSIBLE_VALUE

        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index):
        """Get the prefix sum up to index. Index is 1-based."""
        # Ensure index is within bounds [0, size]
        # Query for value <= t uses index t + 1.
        # If t is -1 (like in the first iteration threshold), index is 0. query(0) should be 0.
        if index < 1:
            return 0
        if index > self.size:
             index = self.size # Querying beyond max index is same as querying max index

        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & (-index)
        return total

# Read input
N = int(input())
A = list(map(int, input().split()))

# Maximum coordinate value for the Fenwick tree
# Values added to FT are stones_after_receiving[y] + y
# stones_after_receiving[y] = A[y] + R_y
# R_y is count of j < y such that stones_after_receiving[j] + j > y - 1
# Max R_y is y.
# Max stones_after_receiving[y] <= max(A) + y.
# Max value added to FT <= max(A) + y + y = max(A) + 2*y.
# Max y is N-1.
# Max value <= max(A) + 2 * (N-1).
# max(A) <= 500000, N <= 500000.
# Max value <= 500000 + 2 * (500000 - 1) = 500000 + 999998 = 1499998.
# So, values are in range [0, 1499998].
# FT needs indices 1 to 1499998 + 1 = 1499999. Size 1499999.
MAX_POSSIBLE_VALUE = 1499998 # The highest possible value added to FT
ft_size = MAX_POSSIBLE_VALUE + 1

ft = FenwickTree(ft_size)

# Array to store the number of stones each alien has
# after receiving gifts when they become adult.
# Corresponds to A[i] + R_{i+1} for alien index i.
stones_after_receiving = [0] * N

# Simulate year by year (0-indexed years 0 to N-1)
for y in range(N):
    # Alien y (index y) becomes adult. Adults are 0 to y-1.
    # Calculate received stones for Alien y
    # R_y = sum(1 for j=0 to y-1 if stones_after_receiving[j] + j > y - 1)
    # R_y = count of values `stones_after_receiving[j] + j` in FT (for j=0 to y-1) that are > (y - 1)
    
    # We need to count values v_j = stones_after_receiving[j] + j for j < y
    # such that v_j > y - 1, which is equivalent to v_j >= y.
    # Count >= y is total count - count < y
    # Count < y is count <= y - 1
    # In FT, query(k) gives count <= k-1 (if using 0-indexed value to 1-indexed index k)
    # Or query(k) gives count <= k (if using value to index value + 1).
    # We map value `v` to index `v + 1`. So query(idx) sums counts for values v <= idx - 1.
    # Count <= t means count for values v where v <= t. These are at indices v+1 <= t+1.
    # Count <= t is ft.query(t + 1).
    # We need count of values >= y. Total count - count < y.
    # Count < y is count <= y - 1. This is ft.query((y - 1) + 1) = ft.query(y).
    
    # Total number of elements added to the FT so far is exactly `y`.
    # The sum of counts for all values currently in the FT should be `y`.
    # ft.query(ft_size) should return `y`.
    
    received_at_y = y - ft.query(y)

    # Update stones for Alien y (after receiving gifts)
    stones_after_receiving[y] = A[y] + received_at_y

    # Add stones_after_receiving[y] + y to the Fenwick tree for future queries
    value_to_add = stones_after_receiving[y] + y
    # Value must be within [0, MAX_POSSIBLE_VALUE] to fit in FT.
    # Based on derivation, value_to_add <= 1499998, so it fits.
    ft.update(value_to_add + 1, 1) # FT is 1-indexed

# Calculate final stone counts B
B = [0] * N
for i in range(N):
    # Alien i (index i) has stones_after_receiving[i] stones after receiving gifts at year i.
    # It gives stones in years i+1, i+2, ..., N-1. There are N-1-i such years.
    # In each such year k, it gives 1 stone if its current count > 0.
    # The number of stones it gives is min(current_count, 1) in each step, where current_count decreases.
    # Total stones given = min(stones_after_receiving[i], number of times it gave).
    # Number of times it gave = number of years k in [i+1, N-1] where count > 0 just before giving.
    # This simplifies to min(stones_after_receiving[i], N - 1 - i) because its count only decreases by 1 per year it gives.
    # It will give a stone every year for N-1-i years unless its count hits 0.
    # The total stones given is min(initial_count_after_receiving, number_of_giving_opportunities).
    # initial_count_after_receiving = stones_after_receiving[i]
    # number_of_giving_opportunities = number of years from i+1 to N-1 = (N-1) - (i+1) + 1 = N - 1 - i.

    # For i = N-1, number of giving opportunities is N-1-(N-1) = 0. min(...) = 0.
    # For i < N-1, number of giving opportunities is N-1-i > 0.
    
    num_years_gives = N - 1 - i
    stones_given = min(stones_after_receiving[i], num_years_gives)

    B[i] = stones_after_receiving[i] - stones_given

# Print output
sys.stdout.write(' '.join(map(str, B)) + '
')