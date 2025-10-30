# YOUR CODE HERE
import sys
import threading

def main():
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    visited = [[False]*W for _ in range(H)]
    directions = [(-1, -1), (-1, 0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                stack = [(i,j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                count +=1
    print(count)
threading.Thread(target=main).start()