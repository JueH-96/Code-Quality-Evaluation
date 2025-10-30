# YOUR CODE HERE
import sys

# Function definitions for faster I/O
def get_ints():
    """Reads a line of space-separated integers."""
    # Reads a line from standard input, removes leading/trailing whitespace,
    # splits the line into strings based on whitespace, and maps each string to an integer.
    return map(int, sys.stdin.readline().strip().split())

def get_list():
    """Reads a line of space-separated integers and returns as a list."""
    # Similar to get_ints(), but returns a list of integers.
    return list(map(int, sys.stdin.readline().strip().split()))

def solve():
    """Solves the main problem."""
    N, K = get_ints() # Read N (length of sequence) and K (subset size requirement)
    A = get_list() # Read the sequence A as a list of integers

    # Use the maximum possible value for A_i from constraints for array sizes.
    # The constraints state A_i <= 10^6.
    # We need arrays indexed up to this maximum value.
    MAX_A_VAL = 1000000 
    M = MAX_A_VAL # M represents the maximum value any element in A can take.

    # Calculate frequencies of each number present in A.
    # freq[x] will store the count of occurrences of the number x in the sequence A.
    # The array size is M+1 to accommodate indices from 0 to M.
    freq = [0] * (M + 1)
    for x in A:
        # Constraints state A_i >= 1. We check if x is within the valid range [1, M].
        # Values outside this range specified by constraints are not expected, but this check adds robustness.
        if 1 <= x <= M:
            freq[x] += 1

    # Calculate C[d]: the count of numbers in the input sequence A
    # that are multiples of d.
    # C[d] = sum(freq[k*d] for k such that k*d <= M)
    # Initialize count array C with zeros. Size M+1 for indices 0 to M.
    C = [0] * (M + 1)
    # Iterate through all possible divisors d from 1 to M.
    for d in range(1, M + 1):
        # Iterate through all multiples of d up to M.
        # The loop `range(d, M + 1, d)` generates d, 2d, 3d, ..., up to the largest multiple <= M.
        for multiple in range(d, M + 1, d):
            # Add the frequency of this multiple to the count for divisor d.
            # If freq[multiple] is 0, adding it does nothing.
            # This correctly computes the total count of elements in A that are multiples of d.
            C[d] += freq[multiple]
    
    # Calculate max_divisor_satisfying_K[g]: This array will store the final answer
    # for any number g. Specifically, it stores the maximum divisor d of g
    # such that there are at least K multiples of d in the input array A (i.e., C[d] >= K).
    # Initialize with 0.
    max_divisor_satisfying_K = [0] * (M + 1) 
    
    # The problem implies K >= 1. Constraints state K <= N.
    # C[1] = N (total number of elements). Since K <= N, C[1] >= K always holds.
    # Thus, d=1 is always a valid divisor satisfying the condition.
    # The logic below ensures that max_divisor_satisfying_K[g] will be updated to at least 1
    # for all g >= 1 that appear in A (since 1 divides every positive integer).

    # Iterate through all potential divisors d from 1 to M.
    for d in range(1, M + 1):
        # Check if d is a valid candidate for GCD based on the K constraint.
        # We need at least K elements in A that are multiples of d.
        if C[d] >= K:
            # If d is valid, it serves as a potential answer (maximum GCD)
            # for all numbers g that are multiples of d.
            # We update the maximum valid divisor found so far for each such g.
            for g in range(d, M + 1, d):
                 # Since we iterate d from 1 upwards, `max()` ensures we store
                 # the largest valid divisor d found so far for g.
                 # If multiple valid divisors d1, d2, ... divide g, this finds the max.
                 max_divisor_satisfying_K[g] = max(max_divisor_satisfying_K[g], d)

    # Prepare the list of results for each element in the original sequence A.
    output = []
    for x in A:
        # The answer for an element A_i is the value computed for g = A_i.
        # This value represents the largest divisor d of A_i such that C[d] >= K.
        # Convert the result to string for efficient joining later.
        output.append(str(max_divisor_satisfying_K[x])) 

    # Print the results, one per line.
    # Using '
'.join is generally faster than printing elements in a loop for large N.
    print('
'.join(output))

# Execute the solve function when the script is run.
solve()