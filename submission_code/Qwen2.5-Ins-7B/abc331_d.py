# YOUR CODE HERE
import sys

input = sys.stdin.read
N, Q = map(int, input().split())

grid = []
for _ in range(N):
    grid.append(input().strip())

def count_black_in_block(i, j):
    return grid[i % N][j % N] == 'B'

def solve(A, B, C, D):
    count = 0
    for i in range(A, C + 1):
        for j in range(B, D + 1):
            if count_black_in_block(i, j):
                count += 1
    return count

queries = []
for _ in range(Q):
    A, B, C, D = map(int, input().split())
    queries.append((A, B, C, D))

for A, B, C, D in queries:
    print(solve(A, B, C, D))