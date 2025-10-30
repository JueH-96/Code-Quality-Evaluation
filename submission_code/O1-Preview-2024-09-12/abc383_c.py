# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import collections
    sys.setrecursionlimit(1 << 25)

    H,W,D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    visited = [[False]*W for _ in range(H)]
    queue = collections.deque()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                visited[i][j] = True
                queue.append((i,j,0))
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while queue:
        i,j,d = queue.popleft()
        if d >= D:
            continue
        for di,dj in dirs:
            ni,nj = i+di, j+dj
            if 0<=ni<H and 0<=nj<W and not visited[ni][nj] and grid[ni][nj]!='#':
                visited[ni][nj] = True
                queue.append((ni,nj,d+1))
    count = 0
    for i in range(H):
        for j in range(W):
            if visited[i][j] and grid[i][j]!='#':
                count +=1
    print(count)


threading.Thread(target=main).start()