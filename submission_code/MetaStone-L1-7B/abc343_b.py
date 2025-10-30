n = int(input())
matrix = []
for _ in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)

for i in range(n):
    connected = []
    for j in range(n):
        if matrix[i][j] == 1 and j != i:
            connected.append(j + 1)
    connected.sort()
    print(' '.join(map(str, connected)))