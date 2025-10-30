N = int(input())
A = list(map(int, input().split()))

# Create a mapping from element to its current position
pos = {}
for i in range(N):
    pos[A[i]] = i

operations = []

for i in range(N):
    if A[i] != i + 1:
        # Find where element (i+1) is currently located
        j = pos[i + 1]
        
        # Update the position mapping
        pos[A[i]] = j
        pos[A[j]] = i
        
        # Swap positions i and j
        A[i], A[j] = A[j], A[i]
        operations.append((i + 1, j + 1))  # Convert to 1-indexed for output

print(len(operations))
for op in operations:
    print(op[0], op[1])