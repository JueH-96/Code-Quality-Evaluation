# Read the input
N = int(input())
P = list(map(int, input().split()))

# Initialize the array A
A = []

# Perform the operations
for i in range(1, N+1):
    A = A[:P[i-1]] + [i] + A[P[i-1]:]

# Print the final array A
print(" ".join(map(str, A)))