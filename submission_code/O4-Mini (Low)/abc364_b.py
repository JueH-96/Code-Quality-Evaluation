def main():
    import sys
    input = sys.stdin.readline
    
    H, W = map(int, input().split())
    si, sj = map(int, input().split())
    # Convert to 0-based indices
    i, j = si - 1, sj - 1

    grid = [list(input().rstrip()) for _ in range(H)]
    X = input().rstrip()

    for c in X:
        ni, nj = i, j
        if c == 'L':
            nj = j - 1
        elif c == 'R':
            nj = j + 1
        elif c == 'U':
            ni = i - 1
        elif c == 'D':
            ni = i + 1

        # Check bounds and emptiness
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
            i, j = ni, nj

    # Convert back to 1-based for output
    print(i + 1, j + 1)

if __name__ == "__main__":
    main()