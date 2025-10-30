import sys

# Define the value of pi to the 100-th decimal place
pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read the input from stdin
N = int(sys.stdin.readline().strip())

# Truncate the value of pi to N decimal places
truncated_pi = pi[:3 + N]

# Print the result
print(truncated_pi)