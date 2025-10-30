# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem of finding the N-th smallest integer that is a sum of exactly three repunits.
    """
    # Read the input integer N from standard input
    n = int(sys.stdin.readline())

    # A repunit is an integer consisting of only digit 1 (e.g., 1, 11, 111).
    # We need to find the N-th smallest value of R_i + R_j + R_k, where R_x is the x-th repunit.
    
    # We need to determine how many repunits are needed.
    # The number of combinations of 3 repunits chosen from the first L repunits (with replacement)
    # is given by combinations with repetition formula: C(L+3-1, 3) = C(L+2, 3) = (L+2)(L+1)L / 6.
    # We need this count to be at least N. The maximum N is 333.
    # If L=11, C(13, 3) = (13 * 12 * 11) / 6 = 286. Not enough.
    # If L=12, C(14, 3) = (14 * 13 * 12) / 6 = 364. This is sufficient.
    # So, we need to consider repunits up to R_12.

    repunits = []
    current_repunit = 0
    num_repunits = 12  # L=12 is sufficient

    # Generate the first 12 repunits
    for _ in range(num_repunits):
        current_repunit = current_repunit * 10 + 1
        repunits.append(current_repunit)
    # Example: repunits[0] = R_1 = 1
    #          repunits[1] = R_2 = 11
    #          ...
    #          repunits[11] = R_12 = 111111111111

    # Generate all possible sums of three repunits.
    # To avoid duplicates and ensure each sum combination R_i + R_j + R_k is generated once,
    # we iterate with indices i <= j <= k.
    # It can be shown that each unique triplet (i, j, k) with i <= j <= k generates a unique sum.
    sums = []
    for i in range(num_repunits):  # Index for the first repunit
        for j in range(i, num_repunits):  # Index for the second repunit (j >= i)
            for k in range(j, num_repunits):  # Index for the third repunit (k >= j)
                # Calculate the sum and add it to the list
                sum_val = repunits[i] + repunits[j] + repunits[k]
                sums.append(sum_val)

    # Sort the list of sums in ascending order
    sums.sort()

    # The N-th smallest sum is located at index N-1 in the sorted list (0-based indexing)
    print(sums[n - 1])

# Execute the solve function to run the program logic
solve()