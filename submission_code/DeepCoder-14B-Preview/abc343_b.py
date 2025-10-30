n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
for i in range(1, n + 1):
    connected = []
    for j in range(n):
        if matrix[i-1][j] == 1:
            connected.append(j + 1)
    connected.sort()
    print(' '.join(map(str, connected)))