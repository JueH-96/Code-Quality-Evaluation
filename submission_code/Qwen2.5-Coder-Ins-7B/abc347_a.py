# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Extract N, K, and the sequence A
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Extract elements that are multiples of K and divide them by K
result = [a // K for a in A if a % K == 0]

# Print the result in ascending order with spaces in between
print(" ".join(map(str, result)))