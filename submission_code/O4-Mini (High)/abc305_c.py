import sys

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]

    # Determine which rows and columns contain at least one '#'
    row_has = [False] * H
    col_has = [False] * W

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                row_has[i] = True
                col_has[j] = True

    # a = first row with a '#', b = last row with a '#'
    for i in range(H):
        if row_has[i]:
            a = i + 1
            break
    for i in range(H - 1, -1, -1):
        if row_has[i]:
            b = i + 1
            break

    # c = first column with a '#', d = last column with a '#'
    for j in range(W):
        if col_has[j]:
            c = j + 1
            break
    for j in range(W - 1, -1, -1):
        if col_has[j]:
            d = j + 1
            break

    # In the rectangle [a..b] x [c..d], find the single missing cookie ('.')
    for i in range(a - 1, b):
        for j in range(c - 1, d):
            if grid[i][j] == '.':
                print(i + 1, j + 1)
                return

if __name__ == "__main__":
    main()