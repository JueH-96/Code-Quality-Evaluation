n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
for i in range(n):
    connected = [j + 1 for j in range(n) if matrix[i][j] == 1]
    print(' '.join(map(str, sorted(connected))))