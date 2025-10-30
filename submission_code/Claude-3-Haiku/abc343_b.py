# Read the input
N = int(input())
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))

# Solve the problem
for i in range(N):
    neighbors = []
    for j in range(N):
        if A[i][j] == 1:
            neighbors.append(j+1)
    print(" ".join(map(str, sorted(neighbors))))