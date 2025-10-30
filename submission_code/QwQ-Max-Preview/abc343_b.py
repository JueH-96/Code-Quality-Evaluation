n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

for row in matrix:
    adjacent = [j + 1 for j in range(n) if row[j] == 1]
    print(' '.join(map(str, adjacent)))