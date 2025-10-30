def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]
    
    # Step 1: Identify magnet cells
    has_magnet = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                has_magnet[i][j] = True
    
    # Step 2: Mark cells adjacent to a magnet (within the grid)
    adj_magnet = [[False]*W for _ in range(H)]
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for i in range(H):
        for j in range(W):
            if has_magnet[i][j]:
                for di, dj in directions:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and not has_magnet[ni][nj]:
                        adj_magnet[ni][nj] = True
    
    # Step 3: Define free cells = empty and not adjacent to magnet
    # blocked cells = empty but adjacent to magnet => degree of freedom = 1
    # magnet cells we ignore for BFS
    free = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if (not has_magnet[i][j]) and (not adj_magnet[i][j]):
                free[i][j] = True
    
    # Step 4: Find connected components among free cells
    visited = [[False]*W for _ in range(H)]
    
    def bfs(start_i, start_j):
        queue = [(start_i, start_j)]
        visited[start_i][start_j] = True
        cnt = 1
        front = 0
        while front < len(queue):
            ci, cj = queue[front]
            front += 1
            for di, dj in directions:
                ni, nj = ci+di, cj+dj
                if 0 <= ni < H and 0 <= nj < W:
                    if free[ni][nj] and not visited[ni][nj]:
                        visited[ni][nj] = True
                        queue.append((ni,nj))
                        cnt += 1
        return cnt
    
    max_component_size = 0
    for i in range(H):
        for j in range(W):
            if free[i][j] and not visited[i][j]:
                size = bfs(i, j)
                if size > max_component_size:
                    max_component_size = size
    
    # Step 5: If no free cell found, result is 1 (since at least one empty cell is guaranteed)
    # Otherwise, result is the largest connected component size.
    if max_component_size == 0:
        print(1)
    else:
        print(max_component_size)

def main():
    solve()

if __name__ == "__main__":
    main()