# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
grid = data[1:]

def count_triples(grid):
    count = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                for k in range(N):
                    if grid[i][k] == 'o' and k != j:
                        for l in range(N):
                            if grid[l][j] == 'o' and l != i and l != k:
                                count += 1
    return count

print(count_triples(grid))