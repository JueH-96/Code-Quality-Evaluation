def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    cubs = [tuple(map(int, input().split())) for _ in range(N)]

    # The coordinates go from 0 to 100, but cells are unit cubes at integer coords 0..99
    D = 100
    size = D * D * D
    grid = [-1] * size

    def idx(x, y, z):
        return x * D * D + y * D + z

    # Fill the grid: assign each unit cube to its cuboid index
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cubs):
        # occupies [x1..x2-1], [y1..y2-1], [z1..z2-1]
        for x in range(x1, x2):
            base_x = x * D * D
            for y in range(y1, y2):
                base_xy = base_x + y * D
                for z in range(z1, z2):
                    grid[base_xy + z] = i

    # Prepare adjacency sets
    neigh = [set() for _ in range(N)]

    # Directions to check: +x, +y, +z
    for x in range(D):
        base_x = x * D * D
        for y in range(D):
            base_xy = base_x + y * D
            for z in range(D):
                i1 = grid[base_xy + z]
                if i1 < 0:
                    continue
                # +x
                if x + 1 < D:
                    j = grid[base_xy + z + D * D]
                    if j >= 0 and j != i1:
                        neigh[i1].add(j)
                        neigh[j].add(i1)
                # +y
                if y + 1 < D:
                    j = grid[base_x + (y + 1) * D + z]
                    if j >= 0 and j != i1:
                        neigh[i1].add(j)
                        neigh[j].add(i1)
                # +z
                if z + 1 < D:
                    j = grid[base_xy + (z + 1)]
                    # Note: z+1 within same base_xy
                    if j >= 0 and j != i1:
                        neigh[i1].add(j)
                        neigh[j].add(i1)

    # Output counts
    out = sys.stdout
    for s in neigh:
        out.write(str(len(s)) + "
")


if __name__ == "__main__":
    main()