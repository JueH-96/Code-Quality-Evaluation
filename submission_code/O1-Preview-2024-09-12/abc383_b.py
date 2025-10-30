# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]

    floor_cells = []
    pos_to_index = {}
    N = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                pos_to_index[(i,j)] = N
                floor_cells.append((i,j))
                N +=1
    if N < 2:
        print(0)
        return
    reach = [0]*N  # reach[u] is a bitmask

    from collections import deque

    for idx_u in range(N):
        i0, j0 = floor_cells[idx_u]
        visited = set()
        queue = deque()
        queue.append((i0, j0, 0))
        while queue:
            i, j, d = queue.popleft()
            if (i,j) in visited or d > D:
                continue
            visited.add((i,j))
            if grid[i][j] == '.':
                idx = pos_to_index[(i,j)]
                reach[idx_u] |= (1<<idx)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+dx, j+dy
                if 0<=ni<H and 0<=nj<W and abs(ni - i0) + abs(nj - j0) <= D:
                    if (ni, nj) not in visited:
                        queue.append((ni,nj,d+1))
    ans = 0
    for u in range(N):
        for v in range(u+1,N):
            total = reach[u] | reach[v]
            count = bin(total).count('1')
            if count > ans:
                ans = count
    print(ans)
    

threading.Thread(target=main).start()