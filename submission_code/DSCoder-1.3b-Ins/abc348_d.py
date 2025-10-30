H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]
N = int(input())
medicines = [list(map(int, input().split())) for _ in range(N)]

# Initialize energy and start point
energy = [0]*H*W
start = (0, 0)

# Set start point and energy
for i, j, e in medicines:
    grid[i-1][j-1] = '.'
    energy[i-1] = e

# Check if Takahashi can reach the goal point
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'T':
            start = (i, j)
            break
    else:
        continue
    break

if start == (0, 0):
    print('No')
else:
    print('Yes')