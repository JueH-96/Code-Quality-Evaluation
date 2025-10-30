# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    grid = input[2:]

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited and grid[nx][ny] == '.':
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited)

    max_degree_of_freedom = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                degree_of_freedom = bfs(i, j)
                max_degree_of_freedom = max(max_degree_of_freedom, degree_of_freedom)

    print(max_degree_of_freedom)

if __name__ == "__main__":
    main()