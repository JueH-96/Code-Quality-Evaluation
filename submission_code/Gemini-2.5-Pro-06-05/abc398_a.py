import sys

# Read the integer N from standard input.
# The .strip() is good practice to handle potential trailing whitespace.
try:
    N = int(sys.stdin.readline().strip())
except (ValueError, IndexError):
    # This handles cases of empty input or non-integer input, though
    # the problem constraints guarantee a valid integer.
    sys.exit(1)

# Determine the central part of the palindrome based on the parity of N.
# If N is even (N % 2 == 0), the center is '=='.
# If N is odd (N % 2 == 1), the center is '='.
center_part = '==' if N % 2 == 0 else '='

# Calculate the number of '-' characters required on each side of the center.
# The total length is N. The length of the dashes is what remains after
# placing the center part, divided by 2 for symmetry.
num_dashes = (N - len(center_part)) // 2

# Construct the final string by concatenating the parts.
# The string multiplication operator '*' is used to create the dash sections.
result = '-' * num_dashes + center_part + '-' * num_dashes

# Print the final result to standard output.
print(result)