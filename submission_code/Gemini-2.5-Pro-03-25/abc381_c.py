# YOUR CODE HERE
import sys

def solve():
    # Read the integer N, the length of the string S
    N = int(sys.stdin.readline())
    # Read the string S
    S = sys.stdin.readline().strip()

    # Constraints state N >= 1 and S contains at least one '/'.
    # No need to handle N=0 or empty S.

    # Compute left_ones array: left_ones[i] stores the length of the block of consecutive '1's ending at index i.
    left_ones = [0] * N
    # Base case for index 0
    # Check if N > 0 (always true by constraints) and if the first character is '1'
    if N > 0 and S[0] == '1':
        left_ones[0] = 1
    # Dynamic programming step: iterate from index 1 to N-1
    for i in range(1, N):
        if S[i] == '1':
            # If the character at index i is '1', the length of consecutive '1's ending here
            # is one more than the length ending at the previous index.
            left_ones[i] = left_ones[i-1] + 1
        # If S[i] is not '1', the block of consecutive '1's is broken.
        # left_ones[i] remains 0 (its initialized value). Explicit else is not needed.

    # Compute right_twos array: right_twos[i] stores the length of the block of consecutive '2's starting at index i.
    right_twos = [0] * N
    # Base case for index N-1
    # Check if N > 0 (always true) and if the last character is '2'
    if N > 0 and S[N-1] == '2':
        right_twos[N-1] = 1
    # Dynamic programming step: iterate backwards from index N-2 down to 0
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            # If the character at index i is '2', the length of consecutive '2's starting here
            # is one more than the length starting at the next index.
            right_twos[i] = right_twos[i+1] + 1
        # If S[i] is not '2', the block of consecutive '2's starting here is of length 0.
        # right_twos[i] remains 0 (its initialized value). Explicit else is not needed.

    # Initialize maximum length found so far. Since S contains at least one '/',
    # the minimum possible length is 1 (for the substring "/"), so initializing to 0 is safe.
    max_len = 0
    
    # Iterate through the string S. Each '/' character is a potential center of an 11/22 substring.
    for i in range(N):
        if S[i] == '/':
            # For a '/' at index i to be the center of an 11/22 string "k ones / k twos":
            # We need k consecutive '1's immediately to its left (ending at index i-1).
            # We need k consecutive '2's immediately to its right (starting at index i+1).

            # Calculate L_i: the length of consecutive '1's ending at index i-1.
            # This information is available in the precomputed left_ones array.
            # Handle boundary case: if i is 0, there are no characters to the left.
            L_i = 0
            if i > 0: 
                L_i = left_ones[i-1]
            
            # Calculate R_i: the length of consecutive '2's starting at index i+1.
            # This information is available in the precomputed right_twos array.
            # Handle boundary case: if i is N-1, there are no characters to the right.
            R_i = 0
            if i < N - 1: 
                R_i = right_twos[i+1]
            
            # The maximum value 'k' for which we have k '1's to the left and k '2's to the right
            # is limited by the minimum of the available lengths L_i and R_i.
            k = min(L_i, R_i)
            
            # The total length of the 11/22 string formed with this 'k' is k (ones) + 1 (slash) + k (twos) = 2*k + 1.
            current_length = 2 * k + 1
            
            # Update the overall maximum length found across all potential '/' centers.
            max_len = max(max_len, current_length)

    # The problem guarantees S contains at least one '/'. The smallest 11/22 string is "/" (k=0),
    # which has length 1. Thus, the final max_len will be at least 1.
    
    # Print the final maximum length to standard output.
    print(max_len)

# Execute the solve function
solve()