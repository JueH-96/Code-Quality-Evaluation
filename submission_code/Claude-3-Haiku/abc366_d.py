import sys

# Read input
N = int(sys.stdin.readline().strip())
A = [[0] * N for _ in range(N)]
for i in range(N):
    A[i] = [int(x) for x in sys.stdin.readline().strip().split()]

Q = int(sys.stdin.readline().strip())
queries = []
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = [int(x) for x in sys.stdin.readline().strip().split()]
    queries.append((Lx, Rx, Ly, Ry, Lz, Rz))

# Process queries
for Lx, Rx, Ly, Ry, Lz, Rz in queries:
    total = 0
    for x in range(Lx, Rx+1):
        for y in range(Ly, Ry+1):
            for z in range(Lz, Rz+1):
                total += A[x-1][y-1][z-1]
    print(total)