# YOUR CODE HERE
N = int(input())
grid = [[False]*101 for _ in range(101)]
for _ in range(N):
    A, B, C, D = map(int, input().split())
    for x in range(A, B):
        for y in range(C, D):
            grid[x][y] = True
S = sum(sum(row) for row in grid)
print(S)