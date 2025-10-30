# YOUR CODE HERE
def count_triples(N, grid):
    count = 0

    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                for k in range(j + 1, N):
                    if grid[i][k] == 'o':
                        for l in range(i + 1, N):
                            if grid[l][j] == 'o' or grid[l][k] == 'o':
                                count += 1

    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = data[1:N+1]

print(count_triples(N, grid))