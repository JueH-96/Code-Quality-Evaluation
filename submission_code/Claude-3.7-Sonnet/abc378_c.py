# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute B
B = [-1] * N
recent_positions = {}

for i in range(N):
    if A[i] in recent_positions:
        B[i] = recent_positions[A[i]]
    recent_positions[A[i]] = i + 1  # Store 1-indexed position

# Print B
print(' '.join(map(str, B)))