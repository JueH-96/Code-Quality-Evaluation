import sys
from collections import deque

def is_possible_to_make_identical(H, W, A, B):
    def is_identical(grid1, grid2):
        for i in range(H):
            for j in range(W):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True

    def swap_rows(grid, i):
        grid[i], grid[i+1] = grid[i+1], grid[i]

    def swap_columns(grid, j):
        for i in range(H):
            grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]

    def bfs(start):
        queue = deque([(start, 0)])
        visited = set()
        visited.add(tuple(map(tuple, start)))

        while queue:
            current, steps = queue.popleft()
            if is_identical(current, B):
                return steps

            for i in range(H-1):
                new_grid = [row[:] for row in current]
                swap_rows(new_grid, i)
                new_state = tuple(map(tuple, new_grid))
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_grid, steps + 1))

            for j in range(W-1):
                new_grid = [row[:] for row in current]
                swap_columns(new_grid, j)
                new_state = tuple(map(tuple, new_grid))
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_grid, steps + 1))

        return -1

    return bfs(A)

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    H = int(data[index])
    W = int(data[index + 1])
    index += 2

    A = []
    for i in range(H):
        row = list(map(int, data[index:index + W]))
        A.append(row)
        index += W

    B = []
    for i in range(H):
        row = list(map(int, data[index:index + W]))
        B.append(row)
        index += W

    result = is_possible_to_make_identical(H, W, A, B)
    print(result)

if __name__ == "__main__":
    main()