# Read input from stdin
import sys

# Read the first line
N, K = map(int, sys.stdin.readline().split())

# Read the second line
A = list(map(int, sys.stdin.readline().split()))

# Extract multiples of K, divide them by K, and store the quotients in a list
quotients = [a // K for a in A if a % K == 0]

# Sort the quotients in ascending order
quotients.sort()

# Print the quotients with spaces in between
print(' '.join(map(str, quotients)))