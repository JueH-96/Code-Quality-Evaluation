n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

for i in range(1, n + 1):
    row = matrix[i - 1]
    connected = [j + 1 for j in range(n) if row[j] == 1]
    connected.sort()
    print(' '.join(map(str, connected)))