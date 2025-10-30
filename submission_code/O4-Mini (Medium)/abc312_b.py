def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    grid = [input().rstrip() for _ in range(N)]

    # Relative positions that must be black
    tl_black = [(r, c) for r in range(3) for c in range(3)]
    br_black = [(r, c) for r in range(6, 9) for c in range(6, 9)]
    # Relative positions that must be white (adjacent to the black 3x3s)
    tl_white = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    br_white = [(5, 5), (5, 6), (5, 7), (5, 8), (6, 5), (7, 5), (8, 5)]

    results = []

    # Try every 9x9 subgrid
    for i in range(N - 8):
        for j in range(M - 8):
            ok = True
            # Check top-left 3x3 blacks
            for dr, dc in tl_black:
                if grid[i + dr][j + dc] != '#':
                    ok = False
                    break
            if not ok:
                continue
            # Check bottom-right 3x3 blacks
            for dr, dc in br_black:
                if grid[i + dr][j + dc] != '#':
                    ok = False
                    break
            if not ok:
                continue
            # Check whites around top-left block
            for dr, dc in tl_white:
                if grid[i + dr][j + dc] != '.':
                    ok = False
                    break
            if not ok:
                continue
            # Check whites around bottom-right block
            for dr, dc in br_white:
                if grid[i + dr][j + dc] != '.':
                    ok = False
                    break
            if not ok:
                continue

            # If all conditions hold, record (1-based) position
            results.append((i + 1, j + 1))

    # Output in lex order
    for x, y in results:
        print(x, y)

if __name__ == "__main__":
    main()