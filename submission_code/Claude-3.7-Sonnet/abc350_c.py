def transform_permutation(N, A):
    operations = []
    
    for i in range(N):
        if A[i] != i + 1:
            # Find the position j where element i+1 is located
            # Start search from position i to ensure i < j
            j = A.index(i + 1, i)
            
            # Swap elements at positions i and j
            A[i], A[j] = A[j], A[i]
            
            # Record the operation (adding 1 for 1-indexing)
            operations.append((i+1, j+1))
    
    return operations

# Read input
N = int(input())
A = list(map(int, input().split()))

# Transform the permutation and get operations
ops = transform_permutation(N, A)

# Output results
print(len(ops))
for i, j in ops:
    print(i, j)