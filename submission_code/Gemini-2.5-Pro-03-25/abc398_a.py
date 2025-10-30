# YOUR CODE HERE
import sys

# Read the integer N from standard input
n = int(sys.stdin.readline())

# Determine the structure of the string based on whether N is odd or even
if n % 2 == 1:
    # N is odd: The string has a single '=' in the middle.
    # The middle index is (n - 1) // 2.
    # The number of '-' characters on each side of the '=' is (n - 1) // 2.
    half = (n - 1) // 2
    result = "-" * half + "=" + "-" * half
else:
    # N is even: The string has "==" in the middle.
    # The two middle indices are n // 2 - 1 and n // 2.
    # The number of '-' characters before "==" is n // 2 - 1.
    # The number of '-' characters after "==" is also n // 2 - 1.
    # Check edge case N=2: half_minus_one = 2//2 - 1 = 0. result = "" + "==" + "" = "==". Correct.
    half_minus_one = n // 2 - 1
    result = "-" * half_minus_one + "==" + "-" * half_minus_one

# Print the resulting string to standard output
print(result)