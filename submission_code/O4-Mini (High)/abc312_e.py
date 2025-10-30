def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    coords = []
    max_x = max_y = max_z = 0
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        coords.append((x1, x2, y1, y2, z1, z2))
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
        if z2 > max_z: max_z = z2

    # grid dimensions
    X = max_x
    Y = max_y
    Z = max_z
    XY = Y * Z
    total_cells = X * XY

    # flatten 3D grid into 1D array, 0 = empty, otherwise stores cuboid id (1-based)
    id_grid = [0] * total_cells

    # fill the grid with cuboid‐ids
    for idx, (x1, x2, y1, y2, z1, z2) in enumerate(coords):
        cid = idx + 1
        for x in range(x1, x2):
            bx = x * XY
            for y in range(y1, y2):
                bxy = bx + y * Z
                for z in range(z1, z2):
                    id_grid[bxy + z] = cid

    # prepare counting
    neighbor_count = [0] * N
    visited = set()
    NN = N
    ig = id_grid
    nc = neighbor_count
    vc = visited

    # check adjacencies in +x direction
    for x in range(X - 1):
        bx = x * XY
        bx1 = bx + XY
        for y in range(Y):
            bxy = bx + y * Z
            bxy1 = bx1 + y * Z
            for z in range(Z):
                i0 = bxy + z
                a = ig[i0]
                if not a: continue
                b = ig[bxy1 + z]
                if not b or a == b: continue
                i, j = a - 1, b - 1
                if i > j: i, j = j, i
                key = i * NN + j
                if key not in vc:
                    vc.add(key)
                    nc[i] += 1
                    nc[j] += 1

    # check adjacencies in +y direction
    for x in range(X):
        bx = x * XY
        for y in range(Y - 1):
            bxy = bx + y * Z
            bxy1 = bx + (y + 1) * Z
            for z in range(Z):
                i0 = bxy + z
                a = ig[i0]
                if not a: continue
                b = ig[bxy1 + z]
                if not b or a == b: continue
                i, j = a - 1, b - 1
                if i > j: i, j = j, i
                key = i * NN + j
                if key not in vc:
                    vc.add(key)
                    nc[i] += 1
                    nc[j] += 1

    # check adjacencies in +z direction
    for x in range(X):
        bx = x * XY
        for y in range(Y):
            bxy = bx + y * Z
            for z in range(Z - 1):
                i0 = bxy + z
                a = ig[i0]
                if not a: continue
                b = ig[i0 + 1]
                if not b or a == b: continue
                i, j = a - 1, b - 1
                if i > j: i, j = j, i
                key = i * NN + j
                if key not in vc:
                    vc.add(key)
                    nc[i] += 1
                    nc[j] += 1

    # output results
    w = sys.stdout.write
    for cnt in nc:
        w(str(cnt) + "
")


if __name__ == "__main__":
    main()