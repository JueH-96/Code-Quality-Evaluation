import sys

# Read N
N = int(sys.stdin.readline())

# Read garbage type parameters
# Store as list of tuples (q, r). Using a list of tuples is sufficient given N <= 100.
# We use 0-indexing internally for the list.
garbage_types = []
for _ in range(N):
    q_i, r_i = map(int, sys.stdin.readline().split())
    garbage_types.append((q_i, r_i))

# Read Q
Q = int(sys.stdin.readline())

# Process queries
for _ in range(Q):
    t_j, d_j = map(int, sys.stdin.readline().split())
    
    # Get q and r for the specified garbage type (t_j is 1-indexed in input)
    # Adjust t_j to 0-indexed access for the garbage_types list
    q = garbage_types[t_j - 1][0]
    r = garbage_types[t_j - 1][1]

    # We need to find the smallest day D >= d_j such that D % q == r.
    # Collection days are of the form k * q + r for some non-negative integer k (k >= 0).
    # We are looking for the smallest non-negative integer k such that:
    # k * q + r >= d_j
    # Rearranging the inequality:
    # k * q >= d_j - r

    # Let diff = d_j - r
    diff = d_j - r

    # We need the smallest non-negative integer k such that k * q >= diff.

    # Case 1: diff <= 0 (i.e., d_j <= r)
    # If diff is non-positive, then 0 * q = 0 is greater than or equal to diff.
    # So, k=0 is the smallest non-negative integer satisfying k * q >= diff.
    # The collection day corresponding to k=0 is 0 * q + r = r.
    # Since d_j <= r and r itself is a collection day (as 0 <= r < q implies r % q == r),
    # r is the first collection day on or after day d_j.
    if diff <= 0: # This condition is equivalent to d_j <= r
        next_collection_day = r
    # Case 2: diff > 0 (i.e., d_j > r)
    # We need k * q >= diff, where diff is positive.
    # Since q > 0, k must also be positive.
    # The smallest integer k satisfying k * q >= diff (for diff > 0, q > 0)
    # is ceil(diff / q).
    # In integer division, ceil(a / b) for positive integers a, b is (a + b - 1) // b.
    # Here a = diff (which is d_j - r > 0) and b = q (> 0).
    else: # This condition is equivalent to d_j > r
        k_min = (diff + q - 1) // q
        # The next collection day is obtained using this minimum k: k_min * q + r
        next_collection_day = k_min * q + r

    # Print the calculated next collection day
    print(next_collection_day)