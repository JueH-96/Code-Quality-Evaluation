# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    Y = int(input[2])
    A = []
    idx = 3
    for i in range(H):
        row = list(map(int, input[idx:idx+W]))
        A.append(row)
        idx += W

    sea_level = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    total_area = H * W
    result = []

    # Create a list of (elevation, row, col) for all cells
    cells = [(A[i][j], i, j) for i in range(H) for j in range(W)]
    cells.sort()

    # Initialize the queue with all cells at sea level
    queue = deque()
    visited = [[False] * W for _ in range(H)]

    for year in range(1, Y + 1):
        sea_level += 1

        # Add newly submerged cells to the queue
        while cells and cells[0][0] <= sea_level:
            _, i, j = cells.pop(0)
            if not visited[i][j]:
                queue.append((i, j))
                visited[i][j] = True
                total_area -= 1

        # Perform BFS to sink connected cells
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and A[nx][ny] <= sea_level:
                    queue.append((nx, ny))
                    visited[nx][ny] = True
                    total_area -= 1

        result.append(total_area)

    for area in result:
        print(area)

if __name__ == "__main__":
    main()