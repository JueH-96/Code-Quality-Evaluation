import heapq

def solve():
    h, w, t = map(int, input().split())
    grid = [list(input()) for _ in range(h)]

    start_row, start_col = -1, -1
    goal_row, goal_col = -1, -1
    candies = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_row, start_col = r, c
            elif grid[r][c] == 'G':
                goal_row, goal_col = r, c
            elif grid[r][c] == 'o':
                candies.append((r, c))

    q = [(0, 0, start_row, start_col, set())]  # (moves, candies, row, col, visited_candies)
    max_candies = -1

    while q:
        moves, num_candies, row, col, visited_candies = heapq.heappop(q)

        if row == goal_row and col == goal_col:
            max_candies = max(max_candies, num_candies)
            continue

        if moves >= t:
            continue

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                new_visited_candies = set(visited_candies)
                if (nr, nc) in candies:
                    new_visited_candies.add((nr, nc))
                heapq.heappush(q, (moves + 1, len(new_visited_candies), nr, nc, new_visited_candies))

    print(max_candies)

solve()