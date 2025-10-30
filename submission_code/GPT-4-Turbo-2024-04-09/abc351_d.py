def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    
    # Directions for moving in the grid (left, right, up, down)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # To check if a cell is blocked by a magnet in its immediate neighborhood
    def is_blocked(i, j):
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    return True
        return False
    
    # To perform a BFS and find the number of reachable cells from (i, j)
    from collections import deque
    
    def bfs(i, j):
        queue = deque([(i, j)])
        visited = set([(i, j)])
        count = 0
        
        while queue:
            ci, cj = queue.popleft()
            count += 1
            for di, dj in directions:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in visited and grid[ni][nj] == '.':
                    if not is_blocked(ni, nj):
                        visited.add((ni, nj))
                        queue.append((ni, nj))
        
        return count
    
    max_degree_of_freedom = 0
    
    # We only start BFS from cells that are not blocked and are not magnets
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not is_blocked(i, j):
                max_degree_of_freedom = max(max_degree_of_freedom, bfs(i, j))
    
    print(max_degree_of_freedom)

if __name__ == "__main__":
    main()