def solve():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())
    # The grid is N x N, N odd, with center = ((N+1)//2 , (N+1)//2).

    # We need to place 'T' in the center cell
    # and then place the dragon parts 1..(N^2 - 1) in a path
    # so that consecutive parts occupy edge-adjacent cells.
    #
    # A convenient way is to create a spiral-like walk around the center,
    # never stepping onto the center itself. This walk visits every cell
    # except the center exactly once, producing a Hamiltonian path on
    # the sub-grid (with center removed). Then we assign the dragon part
    # numbers to the cells in the order they are visited.

    # Coordinates in this solution are 0-based internally; we'll convert to 1-based if needed.
    # But it's simpler just to keep them 0-based for indexing an array of size N x N.

    # Center (0-based) is (mid, mid)
    mid = (N - 1) // 2

    # We will do a "spiral-out" style walk from (mid, mid-1) with an initial direction of "up".
    # Directions in order: up, right, down, left, repeated.
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    dir_index = 0  # start "going up"
    visited = [[False] * N for _ in range(N)]
    # Mark the center as "visited" to skip it
    visited[mid][mid] = True

    # start position
    start_row, start_col = mid, mid - 1
    path = []
    # Check bounds for start
    if not (0 <= start_row < N and 0 <= start_col < N):
        # For N=3, mid=1, mid-1=0 => center=(1,1), start=(1,0) is in range,
        # so it shouldn't be out of range in normal circumstances.
        # But just in case, we can pick a trivial pattern if it's out of range (N=3 corner case).
        # We'll implement a fallback for safety, though typically this won't trigger.
        path = []
        # trivial path for N=3 if that corner case arises
        # Just do something valid:
        # The center is (1,1), place T there, then put 1..8 around in a ring
        # 1 2 3
        # 8 T 4
        # 7 6 5
        print("1 2 3")
        print("8 T 4")
        print("7 6 5")
        return
    visited[start_row][start_col] = True
    path.append((start_row, start_col))

    # We'll keep going until we've visited N^2 - 1 cells.
    # Each step: try to go forward; if blocked or visited, turn right and try again.
    while len(path) < N * N - 1:
        dr, dc = directions[dir_index]
        nr = path[-1][0] + dr
        nc = path[-1][1] + dc
        # check if we can move there
        if 0 <= nr < N and 0 <= nc < N and not visited[nr][nc]:
            # move
            visited[nr][nc] = True
            path.append((nr, nc))
        else:
            # turn right
            dir_index = (dir_index + 1) % 4

    # Now path is a list of (row, col) covering all cells except the center.
    # We'll index them 1..(N^2 - 1).
    # Let's build the output grid:
    grid = [[""] * N for _ in range(N)]
    # Place 'T' in center
    grid[mid][mid] = "T"

    # Fill the path with parts 1..(N^2 - 1)
    for i, (r, c) in enumerate(path, start=1):
        grid[r][c] = str(i)

    # Print the result
    for r in range(N):
        print(" ".join(grid[r]))


def main():
    solve()

if __name__ == "__main__":
    main()