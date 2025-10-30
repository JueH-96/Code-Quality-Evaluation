from collections import deque

def move(p, dx, dy, grid, N):
    x, y = p
    new_x = x + dx
    new_y = y + dy
    if 0 <= new_x < N and 0 <= new_y < N and grid[new_x][new_y] != '#':
        return (new_x, new_y)
    else:
        return p

N = int(input())
grid = [input().strip() for _ in range(N)]

p = []
for i in range(N):
    for j in range(N):
        if grid[i][j] == 'P':
            p.append((i, j))

start1, start2 = p[0], p[1]
initial_state = tuple(sorted([start1, start2]))

visited = set()
queue = deque()
queue.append((initial_state[0], initial_state[1], 0))
visited.add((initial_state[0], initial_state[1]))

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
found = False

while queue:
    current_p1, current_p2, steps = queue.popleft()

    if current_p1 == current_p2:
        print(steps)
        found = True
        break

    for dx, dy in directions:
        new_p1 = move(current_p1, dx, dy, grid, N)
        new_p2 = move(current_p2, dx, dy, grid, N)
        sorted_new = tuple(sorted([new_p1, new_p2]))
        new_state = (sorted_new[0], sorted_new[1])

        if new_state not in visited:
            visited.add(new_state)
            queue.append((new_state[0], new_state[1], steps + 1))

if not found:
    print(-1)