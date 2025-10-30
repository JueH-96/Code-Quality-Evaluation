from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i+2] for i in range(H)]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Precompute is_movable
    is_movable = [[False for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Check all adjacent cells for '#'
                movable = True
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            movable = False
                            break
                if movable:
                    is_movable[i][j] = True
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    max_dof = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and is_movable[i][j] and not visited[i][j]:
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                movable_count = 0
                non_movable_adjacent = set()
                
                while queue:
                    x, y = queue.popleft()
                    movable_count += 1
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#':
                                non_movable_adjacent.add((nx, ny))
                            elif grid[nx][ny] == '.':
                                if is_movable[nx][ny] and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    queue.append((nx, ny))
                                else:
                                    non_movable_adjacent.add((nx, ny))
                
                dof = movable_count + len(non_movable_adjacent)
                if dof > max_dof:
                    max_dof = dof
    
    print(max_dof)

if __name__ == "__main__":
    main()