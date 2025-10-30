def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # Generate a spiral path starting from the center, covering all N*N cells
    mid = N // 2
    # Directions: right, down, left, up
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
    path = []
    x = mid
    y = mid
    path.append((x,y))
    steps = 1
    dir_idx = 0
    total = N * N
    while len(path) < total:
        # for each step length, go in two directions
        for _ in range(2):
            dx, dy = dirs[dir_idx]
            for _ in range(steps):
                x += dx
                y += dy
                # only record if inside the grid
                if 0 <= x < N and 0 <= y < N:
                    path.append((x,y))
                    if len(path) >= total:
                        break
            dir_idx = (dir_idx + 1) & 3
            if len(path) >= total:
                break
        steps += 1

    # Build the output grid
    grid = [[''] * N for _ in range(N)]
    # Place T at the first cell of the path
    cx, cy = path[0]
    grid[cx][cy] = 'T'
    # Place part numbers 1 .. N^2-1 along the rest of the path
    for num in range(1, total):
        rx, ry = path[num]
        grid[rx][ry] = str(num)

    # Print the grid
    out = []
    for i in range(N):
        out.append(" ".join(grid[i]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()