import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # The pattern to follow
    pattern = "snuke"
    
    # Check starting cell
    if grid[0][0] != pattern[0]:
        print("No")
        return
    
    # dist[i][j] = number of steps from (0,0) to (i,j), or -1 if unreachable
    dist = [[-1]*W for _ in range(H)]
    dist[0][0] = 0
    
    dq = deque()
    dq.append((0, 0))
    
    # BFS
    while dq:
        i, j = dq.popleft()
        d = dist[i][j]
        # Try neighbors
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and dist[ni][nj] < 0:
                # Next letter in the pattern
                next_char = pattern[(d+1) % 5]
                if grid[ni][nj] == next_char:
                    dist[ni][nj] = d + 1
                    dq.append((ni, nj))
    
    # Check if (H-1, W-1) was reached
    if dist[H-1][W-1] >= 0:
        print("Yes")
    else:
        print("No")

# Call main
main()