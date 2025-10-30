# YOUR CODE HERE
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Create a dictionary to store indices of each value
value_indices = {}
for idx in range(N):
    if A[idx] not in value_indices:
        value_indices[A[idx]] = []
    value_indices[A[idx]].append(idx)

# Try all pairs (i, j) with i < j
found = False
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        target = X - A[i] - A[j]
        
        # Check if target exists and if there's an index k > j
        if target in value_indices:
            # Binary search or linear search for index > j
            for k_idx in value_indices[target]:
                if k_idx > j:
                    print(i + 1, j + 1, k_idx + 1)  # Convert to 1-indexed
                    found = True
                    break
            if found:
                break
    if found:
        break

if not found:
    print(-1)