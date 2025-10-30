from collections import deque

def main():
    import sys
    from sys import stdin
    input = stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[2 + i] for i in range(H)]
    
    # Precompute can_move
    can_move = [[False for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Check adjacent cells
                adj_magnets = (
                    (i > 0 and grid[i-1][j] == '#') or
                    (i < H-1 and grid[i+1][j] == '#') or
                    (j > 0 and grid[i][j-1] == '#') or
                    (j < W-1 and grid[i][j+1] == '#')
                )
                if not adj_magnets:
                    can_move[i][j] = True
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    max_dof = 0  # degree of freedom
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and can_move[i][j] and not visited[i][j]:
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                component_size = 1
                adjacent_non_movable = set()
                
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                            if can_move[nx][ny] and not visited[nx][ny]:
                                queue.append((nx, ny))
                                visited[nx][ny] = True
                                component_size += 1
                            else:
                                adjacent_non_movable.add((nx, ny))
                
                dof = component_size + len(adjacent_non_movable)
                max_dof = max(max_dof, dof)
    
    # Ensure that at least one empty cell is considered
    print(max_dof)

if __name__ == '__main__':
    main()