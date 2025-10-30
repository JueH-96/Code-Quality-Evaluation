# YOUR CODE HERE
import sys

# Set fast I/O
input = sys.stdin.readline

# Read problem size and initial string
N, Q = map(int, input().split())
S_str = input().strip()

# Create an auxiliary array B where B[i] = 1 if S[i] == S[i+1], else 0.
# A substring S[L..R] is "good" iff the corresponding sum in B is 0.
# We use a Fenwick Tree (BIT) on B for efficient point updates and range sums.
if N > 1:
    B = [1 if S_str[i] == S_str[i+1] else 0 for i in range(N - 1)]
else:
    B = []

# The BIT is 1-indexed for convenience. A size of N is sufficient for array B (size N-1).
bit_size = N
bit = [0] * bit_size

def update(idx, delta):
    """Performs a point update on the BIT (for a 0-based index of B)."""
    idx += 1  # Convert to 1-based index for BIT
    while idx < bit_size:
        bit[idx] += delta
        idx += idx & -idx

def get_prefix_sum(idx):
    """Calculates the prefix sum up to a 0-based index of B using the BIT."""
    idx += 1  # Convert to 1-based index for BIT
    s = 0
    while idx > 0:
        s += bit[idx]
        idx -= idx & -idx
    return s

def range_sum(l, r):
    """Calculates the sum of B[l...r] using the BIT."""
    if l > r:
        return 0
    # Sum(B[l..r]) = PrefixSum(B[0..r]) - PrefixSum(B[0..l-1])
    if l == 0:
        return get_prefix_sum(r)
    return get_prefix_sum(r) - get_prefix_sum(l - 1)

# Initialize the BIT with the initial state of B.
for i in range(len(B)):
    if B[i] == 1:
        update(i, 1)

# Process all queries
output_lines = []
for _ in range(Q):
    query_type, L, R = map(int, input().split())

    if query_type == 1:
        # A flip operation on S[L..R] only affects B at the boundaries.
        # The relationship between adjacent characters inside the flipped range does not change.
        
        # Left boundary: B[L-2] is affected. This is valid only if L > 1.
        if L > 1:
            idx_to_flip = L - 2
            # delta is 1 if B[idx] was 0, and -1 if it was 1.
            delta = 1 - 2 * B[idx_to_flip]
            update(idx_to_flip, delta)
            B[idx_to_flip] = 1 - B[idx_to_flip]

        # Right boundary: B[R-1] is affected. This is valid only if R < N.
        if R < N:
            idx_to_flip = R - 1
            delta = 1 - 2 * B[idx_to_flip]
            update(idx_to_flip, delta)
            B[idx_to_flip] = 1 - B[idx_to_flip]

    else:  # query_type == 2
        # Check if substring S[L..R] is good.
        # This requires checking for bad pairs S[i]==S[i+1] for i in the range [L-1, R-2].
        # This is equivalent to checking if sum(B[L-1...R-2]) == 0.
        l_b = L - 1
        r_b = R - 2
        
        bad_pairs_count = range_sum(l_b, r_b)
        if bad_pairs_count == 0:
            output_lines.append("Yes")
        else:
            output_lines.append("No")

# Print all accumulated results at once for performance.
sys.stdout.write("
".join(output_lines) + "
")