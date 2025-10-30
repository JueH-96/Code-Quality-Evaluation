def main():
    N = int(input().strip())
    grid = [['' for _ in range(N)] for __ in range(N)]
    mid = N // 2
    center = (mid, mid)
    grid[mid][mid] = 'T'
    total = N * N - 1
    num = 1
    left, right, top, bottom = 0, N-1, 0, N-1

    while num <= total:
        for j in range(left, right + 1):
            if (top, j) == center:
                continue
            grid[top][j] = str(num)
            num += 1
            if num > total:
                break
        top += 1
        if num > total:
            break

        for i in range(top, bottom + 1):
            if (i, right) == center:
                continue
            grid[i][right] = str(num)
            num += 1
            if num > total:
                break
        right -= 1
        if num > total:
            break

        for j in range(right, left - 1, -1):
            if (bottom, j) == center:
                continue
            grid[bottom][j] = str(num)
            num += 1
            if num > total:
                break
        bottom -= 1
        if num > total:
            break

        for i in range(bottom, top - 1, -1):
            if (i, left) == center:
                continue
            grid[i][left] = str(num)
            num += 1
            if num > total:
                break
        left += 1

    for row in grid:
        print(" ".join(row))

if __name__ == "__main__":
    main()