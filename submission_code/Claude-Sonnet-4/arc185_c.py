# YOUR CODE HERE
n, x = map(int, input().split())
a = list(map(int, input().split()))

# Create a dictionary to store value -> list of indices (1-based)
value_to_indices = {}
for i in range(n):
    val = a[i]
    if val not in value_to_indices:
        value_to_indices[val] = []
    value_to_indices[val].append(i + 1)  # 1-based indexing

found = False
result = None

# Try all pairs (i, j) where i < j
for i in range(n):
    for j in range(i + 1, n):
        target = x - a[i] - a[j]
        
        # Check if target exists and has an index > j
        if target in value_to_indices:
            for k_idx in value_to_indices[target]:
                if k_idx > j + 1:  # k_idx is 1-based, j is 0-based
                    result = (i + 1, j + 1, k_idx)  # Convert to 1-based
                    found = True
                    break
        
        if found:
            break
    if found:
        break

if found:
    print(result[0], result[1], result[2])
else:
    print(-1)