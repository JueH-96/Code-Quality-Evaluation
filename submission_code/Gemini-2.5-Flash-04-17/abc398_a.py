# YOUR CODE HERE
import sys

# Read input N from standard input
N = int(sys.stdin.readline())

# Initialize an empty string to build the result
result_string = ""

# Determine the structure based on the parity of N
if N % 2 == 1:
    # If N is odd, the string will have a single '=' character at the exact middle.
    # For a string of length N, the middle index is N // 2.
    # The number of '-' characters on each side of the '=' is (N - 1) / 2.
    # Integer division N // 2 correctly calculates (N - 1) / 2 for odd N.
    num_dashes = N // 2
    # Construct the string: dashes on the left + central '=' + dashes on the right
    result_string = '-' * num_dashes + '=' + '-' * num_dashes
else:
    # If N is even, the string will have two adjacent '=' characters in the middle.
    # For a string of length N, the two middle indices are N/2 - 1 and N/2.
    # The number of '-' characters on each side of the '==' block is (N - 2) / 2.
    # Integer division N // 2 - 1 correctly calculates (N - 2) / 2 for even N.
    num_dashes = N // 2 - 1
    # Construct the string: dashes on the left + central '==' + dashes on the right
    result_string = '-' * num_dashes + '==' + '-' * num_dashes

# Print the resulting string to standard output
print(result_string)