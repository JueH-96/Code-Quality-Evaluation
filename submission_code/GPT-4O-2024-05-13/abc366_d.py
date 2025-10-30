# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0

# Read N
N = int(data[index])
index += 1

# Read the 3D array A
A = [[[0] * N for _ in range(N)] for _ in range(N)]
for x in range(N):
    for y in range(N):
        for z in range(N):
            A[x][y][z] = int(data[index])
            index += 1

# Read Q
Q = int(data[index])
index += 1

# Read the queries
queries = []
for _ in range(Q):
    Lx = int(data[index]) - 1
    Rx = int(data[index + 1]) - 1
    Ly = int(data[index + 2]) - 1
    Ry = int(data[index + 3]) - 1
    Lz = int(data[index + 4]) - 1
    Rz = int(data[index + 5]) - 1
    queries.append((Lx, Rx, Ly, Ry, Lz, Rz))
    index += 6

# Process each query and print the result
results = []
for Lx, Rx, Ly, Ry, Lz, Rz in queries:
    total_sum = 0
    for x in range(Lx, Rx + 1):
        for y in range(Ly, Ry + 1):
            for z in range(Lz, Rz + 1):
                total_sum += A[x][y][z]
    results.append(total_sum)

# Print all results
for result in results:
    print(result)