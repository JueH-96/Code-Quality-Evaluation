import sys
import bisect

def main():
    sys.setrecursionlimit(1 << 25)
    H, W, Y = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Collect all cells with their positions and elevations
    cells = []
    for i in range(H):
        for j in range(W):
            cells.append((A[i][j], i, j))

    # Sort cells by elevation
    cells.sort()

    # Union-Find structure
    parent = [i for i in range(H*W + 1)]
    size = [1]*(H*W) + [0]  # size of each set, sea has size 0

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if size[rootX] < size[rootY]:
                parent[rootX] = rootY
                size[rootY] += size[rootX]
            else:
                parent[rootY] = rootX
                size[rootX] += size[rootY]

    # Virtual node for sea
    sea_index = H*W

    submerged_area = 0
    sea_levels = []
    submerged_areas = []

    prev_elevation = -1
    processed = [[False]*W for _ in range(H)]

    idx = 0
    n = H*W
    while idx < n:
        current_elevation = cells[idx][0]
        if current_elevation != prev_elevation:
            if prev_elevation != -1:
                sea_levels.append(prev_elevation)
                submerged_areas.append(submerged_area)
            prev_elevation = current_elevation
        i = cells[idx][1]
        j = cells[idx][2]
        cell_index = i*W + j
        processed[i][j] = True
        # If on perimeter, connect to sea
        if i == 0 or i == H-1 or j == 0 or j == W-1:
            union(cell_index, sea_index)
        # Connect to neighbors with elevation <= current_elevation
        for d in dirs:
            ni, nj = i + d[0], j + d[1]
            if 0 <= ni < H and 0 <= nj < W and processed[ni][nj]:
                neighbor_index = ni*W + nj
                union(cell_index, neighbor_index)
        submerged_area += 1
        # Move to next cell with the same elevation
        while idx < n and cells[idx][0] == current_elevation:
            idx += 1
    # Append the last elevation and submerged area
    sea_levels.append(prev_elevation)
    submerged_areas.append(submerged_area)

    # Now, for each year from 1 to Y, find the submerged area
    # Precompute the submerged area for each year
    for _ in range(Y):
        i = int(sys.stdin.readline())
        # Find the smallest sea_level >= i
        pos = bisect.bisect_left(sea_levels, i)
        if pos < len(sea_levels):
            remaining_area = H*W - submerged_areas[pos]
            print(remaining_area)
        else:
            print(0)

if __name__ == "__main__":
    main()