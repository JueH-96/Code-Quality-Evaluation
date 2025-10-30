# YOUR CODE HERE
N = int(input())
R = input()
C = input()

grid = [['.']*N for _ in range(N)]

for i in range(N):
    if R[i] == 'A':
        grid[i][C.index('A')] = 'A'
    elif R[i] == 'B':
        grid[i][C.index('B')] = 'B'
    else:
        grid[i][C.index('C')] = 'C'

if all(row.count(ch) == 1 for ch in 'ABC' for row in grid) and all(col.count(ch) == 1 for ch in 'ABC' for col in zip(*grid)):
    print('Yes')
    for row in grid:
        print(''.join(row))
else:
    print('No')