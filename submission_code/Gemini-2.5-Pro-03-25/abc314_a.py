# YOUR CODE HERE
import sys

def solve():
    # Read the integer N from standard input
    n = int(sys.stdin.readline())

    # Store the given value of pi to 100 decimal places as a string
    pi_string = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

    # To truncate pi to N decimal places, we need the integer part '3',
    # the decimal point '.', and the first N digits after the decimal point.
    # The total length of the required string is 1 (for '3') + 1 (for '.') + N (for decimal places) = N + 2.

    # Python string slicing s[start:end] extracts characters from index 'start'
    # up to, but not including, index 'end'.
    # We want characters from index 0 up to index (N + 1).
    # Therefore, the slice should be pi_string[0 : N + 2].
    truncated_pi = pi_string[0 : n + 2]

    # Print the truncated string to standard output
    print(truncated_pi)

# Call the solve function to execute the logic
solve()