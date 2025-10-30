def main():
    import sys
    input=sys.stdin.readline
    N, M = map(int, input().split())
    grid = [input().rstrip("
") for _ in range(N)]
    
    # Helper function to check if a 9x9 subgrid at (si, sj) is a valid TaK Code.
    def valid_tak(si, sj):
        # Check top-left 3x3 block: must be black ('#')
        for i in range(3):
            for j in range(3):
                if grid[si + i][sj + j] != '#':
                    return False
        # Check bottom-right 3x3 block: rows 6-8 and cols 6-8 must be black ('#')
        for i in range(6, 9):
            for j in range(6, 9):
                if grid[si + i][sj + j] != '#':
                    return False
        # Check adjacent (neighbors) to top-left block that are inside the 9x9 region should be white ('.')
        # Top-left block covers rows si to si+2, cols sj to sj+2.
        # The adjacent region is the 4x4 block from rows si to si+3 and cols sj to sj+3 minus the black block itself.
        for i in range(0, 4):  # relative indices in the 9x9 block
            for j in range(0, 4):
                # Skip if it's part of the top-left 3x3 black block
                if i < 3 and j < 3:
                    continue
                if grid[si + i][sj + j] != '.':
                    return False
        # Check adjacent (neighbors) to bottom-right block:
        # Bottom-right block covers rows (si+6) to (si+8) and cols (sj+6) to (sj+8).
        # Its adjacent region is rows (si+5) to (si+8) and cols (sj+5) to (sj+8) minus the block.
        for i in range(5, 9):
            for j in range(5, 9):
                if i >= 6 and j >= 6:
                    # part of the black block
                    continue
                if grid[si + i][sj + j] != '.':
                    return False
        # All conditions satisfied.
        return True

    results = []
    for si in range(N - 9 + 1):
        for sj in range(M - 9 + 1):
            if valid_tak(si, sj):
                # Coordinates in output are 1-indexed.
                results.append((si+1, sj+1))
    # Results are generated in lexicographical order from iteration by si then sj.
    for r in results:
        print(r[0], r[1])
    
if __name__ == '__main__':
    main()