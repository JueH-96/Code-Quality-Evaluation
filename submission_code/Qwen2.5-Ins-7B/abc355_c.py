# N T
N, T = map(int, input().split())
# A_1 A_2 ... A_T
A = list(map(int, input().split()))

grid = [[0] * N for _ in range(N)]
rows = [0] * N
cols = [0] * N
diags = [0, 0]

for turn, num in enumerate(A, 1):
    r, c = divmod(num - 1, N)
    grid[r][c] = 1
    rows[r] += 1
    cols[c] += 1
    if r == c:
        diags[0] += 1
    if r + c == N - 1:
        diags[1] += 1
    
    if rows[r] == N or cols[c] == N or diags[0] == N or diags[1] == N:
        print(turn)
        break
else:
    print(-1)