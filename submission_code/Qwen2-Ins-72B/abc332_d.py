from collections import defaultdict
from itertools import permutations

def check(A, B):
    for i in range(H):
        for j in range(W):
            if A[i][j] != B[i][j]:
                return False
    return True

def solve(A, B):
    for perm_row in permutations(range(H)):
        for perm_col in permutations(range(W)):
            grid = [[A[i][j] for j in range(W)] for i in range(H)]
            for i in range(H-1):
                if perm_row[i] > perm_row[i+1]:
                    grid[i], grid[i+1] = grid[i+1], grid[i]
            for j in range(W-1):
                if perm_col[j] > perm_col[j+1]:
                    for i in range(H):
                        grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
            if check(grid, B):
                return sum(perm_row[i] > perm_row[i+1] for i in range(H-1)) + sum(perm_col[j] > perm_col[j+1] for j in range(W-1))
    return -1

H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _ in range(H)]
print(solve(A, B))