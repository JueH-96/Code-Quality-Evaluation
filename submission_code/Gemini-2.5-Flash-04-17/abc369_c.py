import sys

# Function to solve the problem
def solve():
    # Read the number of elements
    N = int(sys.stdin.readline())
    # Read the list of integers
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize the total count of arithmetic progression subarrays.
    # Any subarray of length 1 is an arithmetic progression.
    # There are N such subarrays, corresponding to pairs (i, i) for i from 1 to N in 1-based indexing.
    total_aps = N

    # If N is less than or equal to 1, only length 1 APs exist, which we've counted.
    if N <= 1:
        print(total_aps)
        return

    # Calculate the differences between consecutive elements.
    # differences[i] = A[i+1] - A[i] for 0 <= i < N-1 (using 0-based indexing for A).
    # This `differences` array has length N-1.
    # A subarray A[l...r] (0-based indexing, 0 <= l <= r < N) with length >= 2 is an AP
    # if and only if the sequence of differences A[l+1]-A[l], A[l+2]-A[l+1], ..., A[r]-A[r-1] is constant.
    # In terms of the `differences` array (0-based), this means differences[l...r-1] must be constant.
    # Example: A[0...2] = [A0, A1, A2] is AP if A1-A0 == A2-A1, which corresponds to differences[0] == differences[1].

    differences = [A[i+1] - A[i] for i in range(N - 1)]

    # We iterate through the `differences` array to find contiguous runs of equal values.
    # A contiguous block of `k` identical differences `differences[i...i+k-1]`
    # corresponds to `k * (k + 1) // 2` APs of length >= 2.
    # These are the APs A[p...q] (0-based) where the difference sequence differences[p...q-1]
    # is a sub-sequence of `differences[i...i+k-1]`. This happens when i <= p < q <= i+k.
    # The number of pairs (p, q) satisfying i <= p < q <= i+k is the sum 1 + 2 + ... + k = k*(k+1)//2.

    # Initialize the length of the current run of identical differences.
    # If N > 1, the `differences` array has at least one element (at index 0).
    # This first difference starts a run of length at least 1.
    current_run_length = 1 

    # Iterate through the `differences` array starting from the second element (index 1)
    # up to the last element (index N-2).
    # We compare each element `differences[i]` with the previous one `differences[i-1]`
    # to identify the end of a run of equal differences.
    # The loop range is `range(1, N - 1)`.
    # If N=2, N-1 = 1, range(1, 1) is empty. Loop does not run.
    # If N=3, N-1 = 2, range(1, 2) is i=1. Loop runs once.
    # If N > 3, N-1 > 2, loop runs for i=1, 2, ..., N-2.
    for i in range(1, N - 1):
        if differences[i] == differences[i-1]:
            # The current difference is the same as the previous one, extend the current run.
            current_run_length += 1
        else:
            # The current difference is different from the previous one.
            # This signifies the end of a run of `current_run_length` identical differences
            # that finished at index i-1 in the `differences` array.
            # Add the number of APs of length >= 2 that are derived from this complete run.
            # A run of length k in `differences` contributes k*(k+1)/2 such APs.
            total_aps += current_run_length * (current_run_length + 1) // 2
            
            # Start a new run with the current difference `differences[i]`.
            # This new run begins with a length of 1.
            current_run_length = 1

    # After the loop finishes (which covers indices 1 through N-2 of the differences array),
    # `current_run_length` holds the length of the last run of identical differences
    # ending at index N-2 in the `differences` array.
    # We need to add the contribution from this last run to the total count.
    # This step is executed only if N > 1 (as handled by the initial if condition).
    total_aps += current_run_length * (current_run_length + 1) // 2

    # Print the final total count of arithmetic progression subarrays.
    print(total_aps)

# Call the solve function to run the program
solve()