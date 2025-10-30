# YOUR CODE HERE
import sys

# The value of pi to 100 decimal places
pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read N from stdin
N = int(sys.stdin.readline().strip())

# Print pi to N decimal places
if N == 0:
    print("3")
else:
    print(pi[:N+2])