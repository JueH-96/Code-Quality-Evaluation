# YOUR CODE HERE
import sys

# The value of pi to 100 decimal places
pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read the input
N = int(input().strip())

# Truncate pi to N decimal places
truncated_pi = pi[:2 + N]  # Include the '3.' at the beginning

# Print the result
print(truncated_pi)