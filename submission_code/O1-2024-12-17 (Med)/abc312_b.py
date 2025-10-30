def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2:]

    # Offsets for the black 3x3 blocks:
    # Top-left 3x3 and bottom-right 3x3 within the 9x9 region.
    black_offsets = []
    for r in range(3):
        for c in range(3):
            black_offsets.append((r, c))       # Top-left 3x3
    for r in range(3):
        for c in range(3):
            black_offsets.append((r + 6, c + 6))  # Bottom-right 3x3

    # Offsets for cells that must be white:
    # The 14 cells adjacent (in 8 directions) to either 3x3 black block.
    white_offsets = [
        # Adjacent to top-left 3x3
        (0, 3), (1, 3), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
        # Adjacent to bottom-right 3x3
        (5, 5), (5, 6), (5, 7), (5, 8),
        (6, 5), (7, 5), (8, 5)
    ]

    answers = []
    # Check all possible 9x9 subgrids
    for i in range(N - 8):
        for j in range(M - 8):
            # Verify black cells
            ok = True
            for (r, c) in black_offsets:
                if S[i + r][j + c] != '#':
                    ok = False
                    break
            # Verify white cells
            if ok:
                for (r, c) in white_offsets:
                    if S[i + r][j + c] != '.':
                        ok = False
                        break
            if ok:
                # Store 1-based indices
                answers.append((i + 1, j + 1))

    # Output results in lexicographical (row-major) order
    for r, c in answers:
        print(r, c)

# Do not forget to call main()
if __name__ == "__main__":
    main()