import sys

# The value of pi to the 100th decimal place
pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read the input
N = int(sys.stdin.read().strip())

# Print the value of pi to the N-th decimal place
print(pi[:N+2])