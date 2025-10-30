N = int(input())
A = list(map(int, input().split()))

operations = []

# For each position i, put the correct number in position i
for i in range(N):
    # If current position already has correct number, continue
    if A[i] == i + 1:
        continue
        
    # Find position j that has number i+1
    for j in range(i + 1, N):
        if A[j] == i + 1:
            # Swap elements at positions i and j
            A[i], A[j] = A[j], A[i]
            operations.append((i + 1, j + 1))
            break

# Print number of operations
print(len(operations))

# Print each operation
for op in operations:
    print(op[0], op[1])