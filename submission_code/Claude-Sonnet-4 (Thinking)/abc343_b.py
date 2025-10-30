# Read N
N = int(input())

# Read adjacency matrix
A = []
for i in range(N):
    row = list(map(int, input().split()))
    A.append(row)

# For each vertex i, find connected vertices
for i in range(N):
    connected = []
    for j in range(N):
        if A[i][j] == 1:
            connected.append(j + 1)  # Convert back to 1-indexed
    
    # Print connected vertices
    print(' '.join(map(str, connected)))