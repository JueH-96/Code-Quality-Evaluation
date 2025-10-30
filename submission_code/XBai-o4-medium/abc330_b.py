# Read input
N, L, R = map(int, input().split())
A = list(map(int, input().split()))

# Compute X_i for each element in A
result = [str(max(L, min(R, a))) for a in A]

# Print the result
print(' '.join(result))