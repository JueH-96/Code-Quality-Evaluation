n = int(input())
matrix = []
for _ in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)

for row in matrix:
    connections = []
    for j in range(n):
        if row[j] == 1:
            connections.append(str(j + 1))
    print(' '.join(connections))