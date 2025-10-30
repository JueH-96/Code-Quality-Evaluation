# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
R = data[1]
C = data[2]

# Create a grid of size N x N initialized with '.'
grid = [['.' for _ in range(N)] for _ in range(N)]

# Try to place characters in the grid
for i in range(N):
    for j in range(N):
        if R[i] == 'A' and C[j] == 'A':
            grid[i][j] = 'A'
            R = R[:i] + '.' + R[i+1:]
            C = C[:j] + '.' + C[j+1:]
        elif R[i] == 'B' and C[j] == 'B':
            grid[i][j] = 'B'
            R = R[:i] + '.' + R[i+1:]
            C = C[:j] + '.' + C[j+1:]
        elif R[i] == 'C' and C[j] == 'C':
            grid[i][j] = 'C'
            R = R[:i] + '.' + R[i+1:]
            C = C[:j] + '.' + C[j+1:]

# Check if all characters are placed
if R == '.' * N and C == '.' * N:
    print("Yes")
    for row in grid:
        print(''.join(row))
else:
    print("No")