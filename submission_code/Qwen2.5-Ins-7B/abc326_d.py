# YOUR CODE HERE
from itertools import permutations

def solve(N, R, C):
    chars = 'ABC'
    for perm in permutations(chars * N):
        grid = [''] * N
        for i in range(N):
            grid[i] = ''.join(perm[i*N:(i+1)*N])
        if all(grid[i][i] == R[i] and grid[j].count(C[j]) == 1 for i in range(N) for j in range(N)):
            return grid
    return None

N = int(input())
R = input()
C = input()

result = solve(N, R, C)
if result:
    print('Yes')
    for row in result:
        print(row)
else:
    print('No')