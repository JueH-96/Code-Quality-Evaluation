n = int(input())
gridA = [input() for _ in range(n)]
gridB = [input() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if gridA[i][j] != gridB[i][j]:
            print(i+1, j+1)
            break