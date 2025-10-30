from collections import deque

def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))

    is_sunk = [[False for _ in range(w)] for _ in range(h)]
    remaining_area = h * w

    for year in range(1, y + 1):
        sea_level = year
        will_sink = [[False for _ in range(w)] for _ in range(h)]
        queue = deque()

        for r in range(h):
            for c in range(w):
                if not is_sunk[r][c] and a[r][c] <= sea_level:
                    is_adjacent_sea_or_sunk = False
                    if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                        is_adjacent_sea_or_sunk = True
                    else:
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and is_sunk[nr][nc]:
                                is_adjacent_sea_or_sunk = True
                                break

                    if is_adjacent_sea_or_sunk:
                        will_sink[r][c] = True
                        queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not is_sunk[nr][nc] and not will_sink[nr][nc] and a[nr][nc] <= sea_level:
                    will_sink[nr][nc] = True
                    queue.append((nr, nc))

        newly_sunk_count = 0
        for r in range(h):
            for c in range(w):
                if will_sink[r][c] and not is_sunk[r][c]:
                    newly_sunk_count += 1
                    is_sunk[r][c] = True

        remaining_area -= newly_sunk_count
        print(remaining_area)