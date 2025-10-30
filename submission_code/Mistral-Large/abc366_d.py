import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

A = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            A[x][y][z] = int(data[index])
            index += 1

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, data[index:index + 6])
    index += 6
    queries.append((Lx, Rx, Ly, Ry, Lz, Rz))

prefix_sum = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        for z in range(1, N + 1):
            prefix_sum[x][y][z] = (A[x][y][z] +
                                  prefix_sum[x - 1][y][z] +
                                  prefix_sum[x][y - 1][z] +
                                  prefix_sum[x][y][z - 1] -
                                  prefix_sum[x - 1][y - 1][z] -
                                  prefix_sum[x - 1][y][z - 1] -
                                  prefix_sum[x][y - 1][z - 1] +
                                  prefix_sum[x - 1][y - 1][z - 1])

results = []
for (Lx, Rx, Ly, Ry, Lz, Rz) in queries:
    result = (prefix_sum[Rx][Ry][Rz] -
              prefix_sum[Lx - 1][Ry][Rz] -
              prefix_sum[Rx][Ly - 1][Rz] -
              prefix_sum[Rx][Ry][Lz - 1] +
              prefix_sum[Lx - 1][Ly - 1][Rz] +
              prefix_sum[Lx - 1][Ry][Lz - 1] +
              prefix_sum[Rx][Ly - 1][Lz - 1] -
              prefix_sum[Lx - 1][Ly - 1][Lz - 1])
    results.append(result)

sys.stdout.write("
".join(map(str, results)) + "
")