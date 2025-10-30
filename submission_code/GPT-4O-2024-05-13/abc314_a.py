# YOUR CODE HERE
import sys

# The value of pi to 100 decimal places
pi_100 = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read input
N = int(sys.stdin.read().strip())

# Truncate pi to N decimal places
result = pi_100[:2 + N]

# Print the result
print(result)