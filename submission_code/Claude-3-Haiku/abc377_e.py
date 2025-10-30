# Read the input
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Perform the operations
for _ in range(K):
    new_P = [0] * N
    for i in range(N):
        new_P[i] = P[P[i] - 1]
    P = new_P

# Print the final result
print(" ".join(map(str, P)))