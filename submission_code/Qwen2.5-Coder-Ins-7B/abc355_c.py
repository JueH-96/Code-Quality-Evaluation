# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
A = list(map(int, data[2:]))

grid = [[0] * N for _ in range(N)]
rows = [0] * N
cols = [0] * N
diag1 = 0
diag2 = 0

def mark_cell(num):
    i = (num - 1) // N
    j = (num - 1) % N
    grid[i][j] = 1
    rows[i] += 1
    cols[j] += 1
    if i == j:
        diag1 += 1
    if i + j == N - 1:
        diag2 += 1

for turn in range(1, T + 1):
    mark_cell(A[turn - 1])
    if any(row == N for row in rows) or any(col == N for col in cols) or diag1 == N or diag2 == N:
        print(turn)
        sys.exit(0)

print(-1)