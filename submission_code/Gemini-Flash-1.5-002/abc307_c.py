def solve():
    HA, WA = map(int, input().split())
    A = [input() for _ in range(HA)]
    HB, WB = map(int, input().split())
    B = [input() for _ in range(HB)]
    HX, WX = map(int, input().split())
    X = [input() for _ in range(HX)]

    coords_A = []
    for r in range(HA):
        for c in range(WA):
            if A[r][c] == '#':
                coords_A.append((r, c))

    coords_B = []
    for r in range(HB):
        for c in range(WB):
            if B[r][c] == '#':
                coords_B.append((r, c))

    coords_X = []
    for r in range(HX):
        for c in range(WX):
            if X[r][c] == '#':
                coords_X.append((r,c))

    for offset_A_r in range(100):
        for offset_A_c in range(100):
            for offset_B_r in range(100):
                for offset_B_c in range(100):
                    
                    all_coords = []
                    for r,c in coords_A:
                        all_coords.append((r + offset_A_r, c + offset_A_c))
                    for r,c in coords_B:
                        all_coords.append((r + offset_B_r, c + offset_B_c))

                    min_r = min(coord[0] for coord in all_coords)
                    min_c = min(coord[1] for coord in all_coords)
                    max_r = max(coord[0] for coord in all_coords)
                    max_c = max(coord[1] for coord in all_coords)

                    H = max_r - min_r + 1
                    W = max_c - min_c + 1

                    if H != HX or W != WX:
                        continue

                    result = [['.' for _ in range(W)] for _ in range(H)]
                    for r,c in all_coords:
                        result[r-min_r][c-min_c] = '#'

                    if result == [list(row) for row in X]:
                        print("Yes")
                        return

    print("No")

solve()