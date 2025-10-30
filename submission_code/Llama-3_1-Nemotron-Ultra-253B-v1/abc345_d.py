def main():
    import sys
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    for mask in range(1, 1 << N):
        selected = []
        sum_area = 0
        for i in range(N):
            if (mask >> i) & 1:
                a, b = tiles[i]
                selected.append((a, b))
                sum_area += a * b
        if sum_area != H * W:
            continue

        valid = True
        for a, b in selected:
            can_fit = False
            if a <= H and b <= W:
                can_fit = True
            if b <= H and a <= W:
                can_fit = True
            if not can_fit:
                valid = False
                break
        if not valid:
            continue

        sorted_tiles = []
        for a, b in selected:
            possible = []
            if a <= H and b <= W:
                possible.append((a, b))
            if b <= H and a <= W and (a, b) != (b, a):
                possible.append((b, a))
            possible.sort(key=lambda x: (-max(x), -x[0], -x[1]))
            max_dim = max(max(p[0], p[1]) for p in possible) if possible else 0
            sorted_tiles.append((max_dim, possible))
        sorted_tiles.sort(reverse=True, key=lambda x: x[0])
        tile_orientations = [tile[1] for tile in sorted_tiles]

        grid = [[False] * W for _ in range(H)]

        def backtrack(index):
            if index == len(tile_orientations):
                for row in grid:
                    if not all(row):
                        return False
                return True

            for orient in tile_orientations[index]:
                a, b = orient
                for i in range(H - a + 1):
                    for j in range(W - b + 1):
                        free = True
                        for x in range(i, i + a):
                            for y in range(j, j + b):
                                if grid[x][y]:
                                    free = False
                                    break
                            if not free:
                                break
                        if free:
                            for x in range(i, i + a):
                                for y in range(j, j + b):
                                    grid[x][y] = True
                            if backtrack(index + 1):
                                return True
                            for x in range(i, i + a):
                                for y in range(j, j + b):
                                    grid[x][y] = False
            return False

        if backtrack(0):
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()