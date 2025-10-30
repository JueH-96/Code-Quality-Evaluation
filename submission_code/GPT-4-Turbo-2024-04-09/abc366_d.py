import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

# Read the 3D array
A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            A[x][y][z] = int(data[index])
            index += 1

# Read number of queries
Q = int(data[index])
index += 1

# Read and process each query
results = []
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, data[index:index+6])
    index += 6
    sum_query = 0
    for x in range(Lx, Rx + 1):
        for y in range(Ly, Ry + 1):
            for z in range(Lz, Rz + 1):
                sum_query += A[x][y][z]
    results.append(sum_query)

# Output results
for result in results:
    print(result)