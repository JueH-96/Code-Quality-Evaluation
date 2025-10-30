# YOUR CODE HERE
import sys

def popcount(n):
    """Counts the number of set bits (1s) in an integer."""
    # Using built-in method available from Python 3.10
    # For older versions, use bin(n).count('1') or a bit manipulation loop
    # W <= 18, so bin(n).count('1') is also fast enough.
    # return bin(n).count('1')
    return n.bit_count()

def fwht(a, inverse=False):
    """
    Performs Fast Walsh-Hadamard Transform (FWHT) in-place for XOR convolution.
    The transform is defined by (u, v) -> (u+v, u-v).
    The inverse transform is the same, followed by division by N.
    """
    n = len(a)
    # w = log2(n), number of bits. Equivalent to W in the problem.
    w = n.bit_length() - 1
    
    # Iterative implementation
    for s in range(w):
        length = 1 << s # Current subproblem size / 2
        # Iterate through segments of size 2 * length
        for i in range(0, n, 2 * length):
            # Iterate within the first half of the segment
            for j in range(length):
                u = a[i + j]
                v = a[i + length + j]
                a[i + j] = u + v
                a[i + length + j] = u - v

    # Scale the result for the inverse transform
    if inverse:
        # The division by N needs to happen only after the full transform
        # integer division is appropriate here as sums are integers
        for i in range(n):
            a[i] //= n

# Read input
# Using sys.stdin.readline is faster for large inputs
H, W = map(int, sys.stdin.readline().split())

# N = 2^W, the size of the transform domain (number of possible row configurations)
N = 1 << W

# Count occurrences of each distinct original row configuration (0 to 2^W - 1)
# We use a list of size 2^W initialized to zeros.
counts = [0] * N
for _ in range(H):
    row_str = sys.stdin.readline().strip()
    # Convert the binary string to an integer
    row_int = int(row_str, 2)
    counts[row_int] += 1

# Calculate g[y] = min(popcount(y), W - popcount(y)) for y from 0 to 2^W - 1
# g[y] represents the minimum number of 1s achievable for a single row
# that has configuration y after column flips, by choosing whether to flip that row.
g_values = [0] * N
for y in range(N):
    pc = popcount(y)
    g_values[y] = min(pc, W - pc)

# Compute FWHT of the counts array and the g_values array
# Create copies as FWHT modifies the list in-place
counts_hat = counts[:]
g_hat = g_values[:]

fwht(counts_hat)
fwht(g_hat)

# According to the XOR convolution theorem, FWHT(A *xor* B) = FWHT(A) * FWHT(B)
# where * is element-wise multiplication.
# We want to compute f = counts *xor* g, where f[mask] is the total sum
# for a given column flip mask.
# So, FWHT(f)[k] = FWHT(counts)[k] * FWHT(g)[k].
f_hat = [counts_hat[k] * g_hat[k] for k in range(N)]

# To get f, we compute the Inverse FWHT of f_hat.
# Inverse FWHT is the same transform as forward FWHT, followed by scaling by 1/N.
results = f_hat[:] # Create a copy for inverse transform
fwht(results, inverse=True)

# The results array now contains f[k] for k from 0 to N-1.
# results[k] is the total minimum sum achievable when applying column flip mask k,
# and then optimally flipping rows.
# We need the minimum value among all possible column flip masks.
min_total_sum = min(results)

# Print the minimum total sum
print(min_total_sum)