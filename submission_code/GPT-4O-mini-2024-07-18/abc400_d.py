from collections import deque

def min_front_kicks(H, W, grid, start, end):
    # Directions for moving and kicking
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    kick_effect = [(0, 1), (0, 2), (1, 0), (2, 0), (0, -1), (0, -2), (-1, 0), (-2, 0)]  # kick effects

    # BFS setup
    queue = deque()
    visited = set()
    queue.append((start[0], start[1], 0))  # (row, col, kicks)
    visited.add((start[0], start[1]))

    while queue:
        x, y, kicks = queue.popleft()

        # Check if we reached the destination
        if (x, y) == (end[0], end[1]):
            return kicks

        # Try moving to adjacent cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, kicks))

        # Try kicking in each direction
        for dx, dy in directions:
            # Create a new grid with the kick effect
            new_grid = [list(row) for row in grid]
            for k in range(1, 3):  # kick effect for 1 and 2 steps
                nx, ny = x + k * dx, y + k * dy
                if 0 <= nx < H and 0 <= ny < W:
                    new_grid[nx][ny] = '.'

            # Perform BFS on the new grid
            new_visited = set(visited)  # Copy visited set
            new_queue = deque([(x, y, kicks + 1)])  # Start from the current position with one more kick
            new_visited.add((x, y))

            while new_queue:
                nx, ny, new_kicks = new_queue.popleft()

                # Check if we reached the destination
                if (nx, ny) == (end[0], end[1]):
                    return new_kicks

                # Try moving to adjacent cells in the new grid
                for ddx, ddy in directions:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < H and 0 <= nny < W and new_grid[nnx][nny] == '.':
                        if (nnx, nny) not in new_visited:
                            new_visited.add((nnx, nny))
                            new_queue.append((nnx, nny, new_kicks))

    return -1  # Should never reach here if inputs are valid

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = data[1:H + 1]
    A, B, C, D = map(int, data[H + 1].split())
    
    # Convert to 0-indexed
    start = (A - 1, B - 1)
    end = (C - 1, D - 1)
    
    result = min_front_kicks(H, W, grid, start, end)
    print(result)

if __name__ == "__main__":
    main()