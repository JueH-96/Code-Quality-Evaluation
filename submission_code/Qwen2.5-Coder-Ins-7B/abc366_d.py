import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = []
index = 1
for i in range(N):
    row = []
    for j in range(N):
        row.append(int(data[index]))
        index += 1
    A.append(row)

Q = int(data[index])
index += 1
results = []
for i in range(Q):
    Lx, Rx, Ly, Ry, Lz, Rz = map(int, data[index:index+6])
    index += 6
    total = 0
    for x in range(Lx-1, Rx):
        for y in range(Ly-1, Ry):
            for z in range(Lz-1, Rz):
                total += A[x][y][z]
    results.append(total)

for result in results:
    print(result)