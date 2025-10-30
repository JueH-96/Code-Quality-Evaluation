from collections import deque

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    grid = input_data[2:]

    # We'll do a BFS from (1,1) in 0-based indices (which corresponds to (2,2) in 1-based)
    # and explore all reachable ice squares by "sliding" in each direction until hitting rock.

    visited = [[False]*M for _ in range(N)]
    start_x, start_y = 1, 1  # 0-based for (2,2)
    visited[start_x][start_y] = True

    queue = deque()
    queue.append((start_x, start_y))

    # Directions: up, down, left, right
    # We'll define them as (dx, dy)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == '.':
                if not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
                # keep sliding
                nx += dx
                ny += dy
            # when we encounter a rock or go out of range, we stop

    # Count how many squares are visited
    ans = 0
    for i in range(N):
        for j in range(M):
            if visited[i][j]:
                ans += 1

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()