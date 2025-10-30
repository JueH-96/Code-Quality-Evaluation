import sys

# Use raw input for speed in competitive programming
def get_ints(): return map(int, sys.stdin.readline().strip().split())
def get_str(): return sys.stdin.readline().strip()

def popcount(n):
    """Calculates the number of set bits (1s) in the binary representation of n."""
    # Using the built-in method which is usually efficient
    return bin(n).count('1')

def fwht(a, inverse=False):
    """
    In-place Fast Walsh-Hadamard Transform for XOR convolution.
    Assumes len(a) is a power of 2.
    """
    n = len(a)
    # w = log2(n)
    w = n.bit_length() - 1

    for length in (1 << k for k in range(w)):
        for i in range(0, n, 2 * length):
            for j in range(length):
                u = a[i + j]
                v = a[i + length + j]
                a[i + j] = u + v
                a[i + length + j] = u - v

    if inverse:
        # Divide by N = 2^W after the inverse transform
        # For integer arithmetic, we can use integer division
        # In the sum/difference formulation, division by 2 happens at each step.
        # Total division by 2^W = N.
        for i in range(n):
            a[i] //= n

def solve():
    H, W = get_ints()

    # Step 1-3: Read grid and count distinct rows
    row_counts = {}
    for _ in range(H):
        row_str = get_str()
        # Convert binary string to integer
        # The first character is treated as the most significant bit
        row_int = int(row_str, 2)
        row_counts[row_int] = row_counts.get(row_int, 0) + 1

    # Step 4: Create vector P (counts of distinct rows)
    N = 1 << W
    P = [0] * N
    for row_int, count in row_counts.items():
        P[row_int] = count

    # Step 5: Create vector Q (cost function min(popcount, W-popcount))
    # For a modified row with k ones (and W-k zeros), the cost is min(k, W-k)
    Q = [0] * N
    for x in range(N):
        pc = popcount(x)
        Q[x] = min(pc, W - pc)

    # Step 6-8: Compute FWHT(P), FWHT(Q), and their element-wise product
    # The sum we want for a mask m is S(m) = sum_{v} P_counts[v] * Q_costs[v ^ m]
    # This sum is the XOR convolution (P * Q)[m], where convolution is defined as
    # (A * B)[k] = sum_i A[i] B[i ^ k]
    # This can be computed efficiently using FWHT: FWHT(P * Q) = FWHT(P) elementwise_product FWHT(Q)

    fwht(P) # P becomes FWHT(P)
    fwht(Q) # Q becomes FWHT(Q)

    # P will store FWHT(C) where C = P * Q
    # Store the product directly in P to save memory
    for i in range(N):
         P[i] *= Q[i]

    # Step 9: Compute Inverse FWHT to get C = P * Q
    # C[m] is the total sum of costs for column mask m
    fwht(P, inverse=True) # P becomes C

    # Step 10: Find the minimum value in C
    # The result of Inverse FWHT should be integers.
    min_total_sum = P[0] # Initialize with the first value
    for m in range(1, N):
        min_total_sum = min(min_total_sum, P[m])

    # Step 11: Print the answer
    print(min_total_sum)

solve()