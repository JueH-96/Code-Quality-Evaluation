import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    grid = data[2:2+H]

    # Precompute, for each cell (only cells without magnets are of interest),
    # whether it is "safe". A cell is safe if none of its four (in-bound)
    # neighbors contain a magnet ('#').
    safe = [[False] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                is_safe = True
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            is_safe = False
                            break
                safe[i][j] = is_safe
            else:
                safe[i][j] = False  # not used for magnet cells

    # We now observe that the movement rule is:
    #   - From an unsafe cell (one that is adjacent to a magnet), you cannot move.
    #   - From a safe cell (no adjacent magnets), you may move to any adjacent cell.
    #
    # Thus, if you start from an unsafe cell your reachable set is just that cell: degree 1.
    # If you start from a safe cell, you can step from safe cell to any adjacent empty cell.
    # Notice that when leaving a safe cell you may land in an unsafe cell.
    # But once you are in an unsafe cell you cannot move further.
    #
    # Hence, for a safe cell, consider the connected component of safe cells (using only safe cells).
    # From any safe cell in that component you can reach:
    #   1) All safe cells in that component.
    #   2) Any unsafe cell that is adjacent to any safe cell in the component.
    #
    # The degree of freedom is the total unique count of such cells.
    # For unsafe starting cells, the reachable set is just {that cell} so degree=1.
    
    visited = [[False] * W for _ in range(H)]
    max_degree = 1  # initialize with 1 since an unsafe cell yields degree 1

    for i in range(H):
        for j in range(W):
            # Process only safe empty cells that have not been visited.
            if grid[i][j] == '.' and safe[i][j] and not visited[i][j]:
                # Begin a BFS to find the connected component of safe cells.
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                safe_count = 0  # count of safe cells in this component
                adjacent_unsafe = set()  # set of unsafe empty cells adjacent to the component

                while q:
                    x, y = q.popleft()
                    safe_count += 1
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + di, y + dj
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                            if safe[nx][ny]:
                                if not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    q.append((nx, ny))
                            else:
                                # This neighbor is empty but unsafe; record it.
                                adjacent_unsafe.add((nx, ny))
                # Total reachable cells when starting from any safe cell in this component.
                candidate = safe_count + len(adjacent_unsafe)
                if candidate > max_degree:
                    max_degree = candidate

    sys.stdout.write(str(max_degree))

if __name__ == '__main__':
    main()