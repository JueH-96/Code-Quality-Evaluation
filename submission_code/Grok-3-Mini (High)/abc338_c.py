import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = [int(data[index + i]) for i in range(N)]
index += N
A = [int(data[index + i]) for i in range(N)]
index += N
B = [int(data[index + i]) for i in range(N)]
index += N

# Compute the maximum possible X
max_X = min(Q[i] // A[i] for i in range(N) if A[i] > 0)

# Initialize the maximum servings
max_servings = 0

# Iterate over all possible X values
for X in range(0, max_X + 1):
    # Compute the maximum Y for this X
    Y_max = min((Q[i] - A[i] * X) // B[i] for i in range(N) if B[i] > 0)
    # Compute current total servings
    current_servings = X + Y_max
    # Update maximum servings if current is larger
    if current_servings > max_servings:
        max_servings = current_servings

# Output the result
print(max_servings)