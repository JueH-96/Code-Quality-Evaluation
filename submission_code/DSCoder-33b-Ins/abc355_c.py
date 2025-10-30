# YOUR CODE HERE

N, T = map(int, input().split())
A = list(map(int, input().split()))

grid = [[0]*N for _ in range(N)]

def check_bingo(grid):
    for i in range(N):
        if sum(grid[i]) == N:
            return True
        if sum(grid[j][i] for j in range(N)) == N:
            return True
    if sum(grid[i][i] for i in range(N)) == N:
        return True
    if sum(grid[i][N-i-1] for i in range(N)) == N:
        return True
    return False

for i in range(T):
    num = A[i]
    row = (num-1) // N
    col = (num-1) % N
    grid[row][col] = 1
    if check_bingo(grid):
        print(i+1)
        break
else:
    print(-1)