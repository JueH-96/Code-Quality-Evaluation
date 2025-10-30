from collections import deque

def bfs(grid, start1, start2):
    n = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(start1, start2, 0)])
    visited = set((start1, start2))

    while queue:
        (x1, y1), (x2, y2), moves = queue.popleft()

        if (x1, y1) == (x2, y2):
            return moves

        for dx, dy in directions:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            if 0 <= nx1 < n and 0 <= ny1 < n and grid[nx1][ny1] != '#':
                if 0 <= nx2 < n and 0 <= ny2 < n and grid[nx2][ny2] != '#':
                    new_state = ((nx1, ny1), (nx2, ny2))
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state[0], new_state[1], moves + 1))
                else:
                    new_state = ((nx1, ny1), (x2, y2))
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state[0], new_state[1], moves + 1))
            else:
                if 0 <= nx2 < n and 0 <= ny2 < n and grid[nx2][ny2] != '#':
                    new_state = ((x1, y1), (nx2, ny2))
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state[0], new_state[1], moves + 1))

    return -1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    grid = data[1:n+1]

    start1 = start2 = None
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                if start1 is None:
                    start1 = (i, j)
                else:
                    start2 = (i, j)

    result = bfs(grid, start1, start2)
    print(result)

if __name__ == "__main__":
    main()