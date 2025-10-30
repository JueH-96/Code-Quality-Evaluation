from collections import deque
import sys

def main():
    H, W, T = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]

    # Find start and goal positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)

    # Find all candy positions
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'o':
                candies.append((i, j))

    # Initialize queue for BFS
    queue = deque([(start, 0, 0, set())])  # (position, moves, candies, visited)

    # Initialize maximum candies
    max_candies = 0

    while queue:
        (x, y), moves, candies_collected, visited = queue.popleft()

        # If we've reached the goal, update max_candies
        if (x, y) == goal:
            max_candies = max(max_candies, candies_collected)

        # If we've exceeded the move limit, skip
        if moves > T:
            continue

        # Explore all possible moves
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            # Check if the move is valid
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                # If we've already visited this cell, skip
                if (nx, ny) in visited:
                    continue

                # Mark the cell as visited
                new_visited = visited.copy()
                new_visited.add((nx, ny))

                # Update candies collected
                new_candies_collected = candies_collected + (1 if (nx, ny) in candies and (nx, ny) not in visited else 0)

                # Add the new state to the queue
                queue.append(((nx, ny), moves + 1, new_candies_collected, new_visited))

    # If we couldn't reach the goal, print -1
    if max_candies == 0 and goal != start:
        print(-1)
    else:
        print(max_candies)

if __name__ == '__main__':
    main()