def main():
    import sys
    input = sys.stdin.readline

    line = input().strip()
    if not line:
        return
    N = int(line)
    S = [input().rstrip('
') for _ in range(N)]

    # Maximum width of the output (number of columns)
    M = max(len(s) for s in S)

    # Prepare an N x M grid filled with '*'
    grid = [['*'] * M for _ in range(N)]

    # Place each S[i] into the grid at row (N-1-i), columns 0..len(S[i])-1
    for i, s in enumerate(S):
        row = N - 1 - i
        for j, ch in enumerate(s):
            grid[row][j] = ch

    # Build and print each column T_j by trimming trailing '*'
    out = []
    for j in range(M):
        # Find the bottommost non-'*' in this column
        last = -1
        for r in range(N):
            if grid[r][j] != '*':
                last = r
        # Collect characters from row 0 to row last (inclusive)
        col_str = ''.join(grid[r][j] for r in range(last + 1))
        out.append(col_str)

    sys.stdout.write("
".join(out))


# Call the main function
main()