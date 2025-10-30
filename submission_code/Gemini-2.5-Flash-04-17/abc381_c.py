import sys

def solve():
    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Precompute number of consecutive '1's ending at index i
    # ones_ending_at_idx[i] stores the length of the longest sequence of '1's
    # that ends at index i.
    ones_ending_at_idx = [0] * N
    for i in range(N):
        if S[i] == '1':
            # If S[i] is '1', the count ending at i is 1 plus the count ending at i-1 (if i > 0)
            ones_ending_at_idx[i] = (ones_ending_at_idx[i-1] if i > 0 else 0) + 1
        else:
            # If S[i] is not '1', the sequence of '1's is broken, so the count is 0
            ones_ending_at_idx[i] = 0

    # Precompute number of consecutive '2's starting at index i
    # twos_starting_at_idx[i] stores the length of the longest sequence of '2's
    # that starts at index i.
    twos_starting_at_idx = [0] * N
    for i in range(N - 1, -1, -1):
        if S[i] == '2':
            # If S[i] is '2', the count starting at i is 1 plus the count starting at i+1 (if i < N - 1)
            twos_starting_at_idx[i] = (twos_starting_at_idx[i+1] if i < N - 1 else 0) + 1
        else:
            # If S[i] is not '2', the sequence of '2's is broken, so the count is 0
            twos_starting_at_idx[i] = 0

    max_len = 0

    # Iterate through potential center '/' characters
    # An 11/22 string must have '/' as its middle character.
    for m in range(N):
        if S[m] == '/':
            # For S[m] to be the center of a 11/22 string of length 2k+1,
            # we need k '1's immediately to the left (at indices m-1, m-2, ..., m-k)
            # and k '2's immediately to the right (at indices m+1, m+2, ..., m+k).

            # The number of consecutive '1's available immediately to the left of m
            # is the count ending at index m-1. This is ones_ending_at_idx[m-1].
            # If m is 0, there are no characters to the left, so available 1s is 0.
            left_potential = ones_ending_at_idx[m-1] if m > 0 else 0

            # The number of consecutive '2's available immediately to the right of m
            # is the count starting at index m+1. This is twos_starting_at_idx[m+1].
            # If m is N-1, there are no characters to the right, so available 2s is 0.
            right_potential = twos_starting_at_idx[m+1] if m < N - 1 else 0

            # The maximum number of pairs (k) of '1's and '2's we can form
            # is limited by the minimum of available '1's and available '2's.
            k = min(left_potential, right_potential)

            # The length of the 11/22 string with k pairs is 2*k + 1 (k ones, /, k twos)
            # k=0 corresponds to length 1 ('/')
            current_max_len = 2 * k + 1

            # Update the overall maximum length found so far
            max_len = max(max_len, current_max_len)

    # The problem guarantees S contains at least one '/'.
    # The smallest 11/22 string is '/', which has length 1.
    # If S has at least one '/', max_len will be updated to at least 1.
    # Initializing max_len = 0 and updating it is correct.

    print(max_len)

solve()