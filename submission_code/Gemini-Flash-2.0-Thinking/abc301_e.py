from collections import deque

def solve():
    h, w, t = map(int, input().split())
    grid = [input() for _ in range(h)]

    start = None
    goal = None
    candies = []
    candy_coords = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
            elif grid[r][c] == 'o':
                candies.append((r, c))

    num_candies = len(candies)
    locations = [start] + candies + [goal]
    num_locations = len(locations)

    def bfs(start_node):
        start_r, start_c = locations[start_node]
        distances = {}
        queue = deque([(start_r, start_c, 0)])
        distances[(start_r, start_c)] = 0

        while queue:
            r, c, dist = queue.popleft()

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#' and (nr, nc) not in distances:
                    distances[(nr, nc)] = dist + 1
                    queue.append((nr, nc, dist + 1))
        return distances

    all_distances = [[float('inf')] * num_locations for _ in range(num_locations)]
    for i in range(num_locations):
        distances = bfs(i)
        for j in range(num_locations):
            if tuple(locations[j]) in distances:
                all_distances[i][j] = distances[tuple(locations[j])]

    dp = {}  # (current_location_index, collected_candies_mask) -> min_time
    queue = deque([(0, 0, 0)])  # (location_index, candies_mask, time)
    dp[(0, 0)] = 0
    max_candies = -1

    while queue:
        current_loc_index, candies_mask, current_time = queue.popleft()

        # Try moving to the goal
        goal_index = num_locations - 1
        dist_to_goal = all_distances[current_loc_index][goal_index]
        if current_time + dist_to_goal <= t:
            max_candies = max(max_candies, bin(candies_mask).count('1'))

        # Try moving to an unvisited candy
        for i in range(num_candies):
            if not (candies_mask & (1 << i)):
                next_candy_index = i + 1
                dist_to_candy = all_distances[current_loc_index][next_candy_index]
                if dist_to_candy != float('inf'):
                    new_time = current_time + dist_to_candy
                    new_mask = candies_mask | (1 << i)
                    if (next_candy_index, new_mask) not in dp or new_time < dp[(next_candy_index, new_mask)]:
                        dp[(next_candy_index, new_mask)] = new_time
                        queue.append((next_candy_index, new_mask, new_time))

    print(max_candies)

solve()