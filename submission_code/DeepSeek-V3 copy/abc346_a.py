# Read input values
N = int(input())
A = list(map(int, input().split()))

# Compute B_i values
B = [A[i] * A[i+1] for i in range(N-1)]

# Print the result
print(' '.join(map(str, B)))