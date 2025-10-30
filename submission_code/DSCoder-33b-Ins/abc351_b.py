# YOUR CODE HERE
N = int(input())
gridA = [input() for _ in range(N)]
gridB = [input() for _ in range(N)]

for i in range(N):
    for j in range(N):
        if gridA[i][j] != gridB[i][j]:
            print(i+1, j+1)
            break