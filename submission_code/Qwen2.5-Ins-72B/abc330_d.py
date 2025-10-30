# YOUR CODE HERE
import sys

def count_triples(N, grid):
    row_counts = [0] * N
    col_counts = [0] * N
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1

    total_triples = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                total_triples += (row_counts[i] - 1) * (col_counts[j] - 1)
    return total_triples

N = int(input())
grid = [input().strip() for _ in range(N)]

print(count_triples(N, grid))