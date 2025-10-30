import sys

# The value of pi to the 100-th decimal place as a string.
# Source: Provided in the problem description.
pi_string = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read the integer N from standard input.
N = int(sys.stdin.readline())

# Truncate the pi string to N decimal places.
# The string starts with '3.' (2 characters) followed by 100 decimal digits.
# To get N decimal places, we need the first 2 characters ('3.') plus the first N decimal digits.
# The slice will start from index 0 and go up to (but not including) index 2 + N.
truncated_pi = pi_string[:2 + N]

# Print the truncated pi value.
print(truncated_pi)