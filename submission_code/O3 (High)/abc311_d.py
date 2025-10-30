import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    grid = [input().strip() for _ in range(N)]

    # zero-based coordinates of the start cell (2,2) in 1-based indexing
    sr, sc = 1, 1

    # visited_rest  – cells where the player can stop (already processed in BFS)
    # touched       – cells the player has ever passed or rested on
    visited_rest = [[False] * M for _ in range(N)]
    touched       = [[False] * M for _ in range(N)]

    q = deque()
    q.append((sr, sc))
    visited_rest[sr][sc] = True
    touched[sr][sc] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    while q:
        r, c = q.popleft()

        for dr, dc in directions:
            nr, nc = r, c

            # Slide in the chosen direction until the next square is a rock
            while True:
                tr, tc = nr + dr, nc + dc
                if grid[tr][tc] == '#':            # rock: stop sliding
                    break
                nr, nc = tr, tc                    # move to the next ice cell
                if not touched[nr][nc]:
                    touched[nr][nc] = True         # mark the cell as touched

            # (nr, nc) is the cell where the player finally stops
            if not visited_rest[nr][nc]:
                visited_rest[nr][nc] = True
                q.append((nr, nc))

    # Count every ice cell that has been touched
    answer = sum(cell for row in touched for cell in row)
    print(answer)


if __name__ == "__main__":
    main()