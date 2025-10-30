import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
L = int(data[1])
R = int(data[2])

# Generate the initial sequence
A = list(range(1, N + 1))

# Reverse the sublist from L-1 to R (0-indexed)
A[L-1:R] = A[L-1:R][::-1]

# Print the resulting sequence
print(" ".join(map(str, A)))