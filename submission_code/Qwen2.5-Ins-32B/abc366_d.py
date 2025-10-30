import sys

def read_ints(): return map(int, sys.stdin.readline().strip().split())

def preprocess(N, A):
    # Preprocess the 3D array to allow fast range sum queries
    for x in range(N):
        for y in range(N):
            for z in range(1, N):
                A[x][y][z] += A[x][y][z-1]
    for x in range(N):
        for y in range(1, N):
            for z in range(N):
                A[x][y][z] += A[x][y-1][z]
    for x in range(1, N):
        for y in range(N):
            for z in range(N):
                A[x][y][z] += A[x-1][y][z]

def query(A, Lx, Rx, Ly, Ry, Lz, Rz):
    # Calculate the sum in the given range using the preprocessed array
    result = A[Rx][Ry][Rz]
    if Lx > 0:
        result -= A[Lx-1][Ry][Rz]
    if Ly > 0:
        result -= A[Rx][Ly-1][Rz]
    if Lz > 0:
        result -= A[Rx][Ry][Lz-1]
    if Lx > 0 and Ly > 0:
        result += A[Lx-1][Ly-1][Rz]
    if Lx > 0 and Lz > 0:
        result += A[Lx-1][Ry][Lz-1]
    if Ly > 0 and Lz > 0:
        result += A[Rx][Ly-1][Lz-1]
    if Lx > 0 and Ly > 0 and Lz > 0:
        result -= A[Lx-1][Ly-1][Lz-1]
    return result

N = int(input())
A = [list(list(read_ints()) for _ in range(N)) for _ in range(N)]
preprocess(N, A)

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = read_ints()
    print(query(A, Lx-1, Rx-1, Ly-1, Ry-1, Lz-1, Rz-1))