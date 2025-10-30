from collections import deque

def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]

    humidifiers = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifiers.append((r, c))

    humidified_floor_cells = set()
    queue = deque([(r, c, 0) for r, c in humidifiers])
    visited = set(humidifiers)

    while queue:
        r, c, dist = queue.popleft()

        if grid[r][c] == '.':
            humidified_floor_cells.add((r, c))
        elif grid[r][c] == 'H':
            humidified_floor_cells.add((r,c))

        if dist < d:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h && 0 <= nc < w && grid[nr][nc] != '#' && (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    print(len(humidified_floor_cells))

if __name__ == "__main__":
    solve()