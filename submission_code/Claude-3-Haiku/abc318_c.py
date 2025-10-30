import math

# Read input
N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Calculate the minimum cost
total_cost = 0
passes_used = 0

# Buy as many batches of passes as needed
while passes_used < N:
    num_passes = min(N - passes_used, D)
    total_cost += P
    passes_used += num_passes

# Pay regular fare for remaining days
for i in range(N):
    if passes_used > 0:
        passes_used -= 1
    else:
        total_cost += F[i]

# Print the answer
print(total_cost)