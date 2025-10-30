import sys
input = sys.stdin.read
data = input().split()

idx = 0
N = int(data[idx])
idx += 1

A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            A[x][y][z] = int(data[idx])
            idx += 1

Q = int(data[idx])
idx += 1

queries = []
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, data[idx:idx + 6])
    idx += 6
    queries.append((Lx, Rx, Ly, Ry, Lz, Rz))

# Prefix sum calculation
prefix_sum = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            prefix_sum[x][y][z] = A[x][y][z] + prefix_sum[x-1][y][z] + prefix_sum[x][y-1][z] + prefix_sum[x][y][z-1] - prefix_sum[x-1][y-1][z] - prefix_sum[x-1][y][z-1] - prefix_sum[x][y-1][z-1] + prefix_sum[x-1][y-1][z-1]

# Function to calculate the sum for a given query
def query_sum(Lx, Rx, Ly, Ry, Lz, Rz):
    return (prefix_sum[Rx][Ry][Rz] - prefix_sum[Lx-1][Ry][Rz] - prefix_sum[Rx][Ly-1][Rz] - prefix_sum[Rx][Ry][Lz-1] +
            prefix_sum[Lx-1][Ly-1][Rz] + prefix_sum[Lx-1][Ry][Lz-1] + prefix_sum[Rx][Ly-1][Lz-1] - prefix_sum[Lx-1][Ly-1][Lz-1])

# Process each query
results = []
for Lx, Rx, Ly, Ry, Lz, Rz in queries:
    result = query_sum(Lx, Rx, Ly, Ry, Lz, Rz)
    results.append(result)

# Output the results
for result in results:
    print(result)