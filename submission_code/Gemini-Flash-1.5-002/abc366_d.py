# YOUR CODE HERE
N = int(input())
A = []
for _ in range(N * N):
    A.extend(list(map(int, input().split())))
A = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(N)]
k = 0
for i in range(N):
    for j in range(N):
        for l in range(N):
            A[i][j][l] = int(A[k])
            k += 1

Q = int(input())
for _ in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
    total_sum = 0
    for x in range(Lx - 1, Rx):
        for y in range(Ly - 1, Ry):
            for z in range(Lz - 1, Rz):
                total_sum += A[x][y][z]
    print(total_sum)