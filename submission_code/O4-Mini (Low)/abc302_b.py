def main():
    import sys
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]

    # Offsets for 8 possible directions (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),          (0,  1),
                  (1,  -1), (1,  0), (1,  1)]
    target = "snuke"

    for i in range(H):
        for j in range(W):
            # looking for 's' at A1
            if grid[i][j] != 's':
                continue
            # try each direction
            for dr, dc in directions:
                coords = []
                ok = True
                for k in range(5):
                    ni = i + dr * k
                    nj = j + dc * k
                    # check bounds
                    if not (0 <= ni < H and 0 <= nj < W):
                        ok = False
                        break
                    # check letter
                    if grid[ni][nj] != target[k]:
                        ok = False
                        break
                    coords.append((ni + 1, nj + 1))  # convert to 1-based
                if ok:
                    # print result
                    out = sys.stdout
                    for r, c in coords:
                        out.write(f"{r} {c}
")
                    return

if __name__ == "__main__":
    main()