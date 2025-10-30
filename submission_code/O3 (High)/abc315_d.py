import sys


def main() -> None:
    rd = sys.stdin.readline

    H, W = map(int, rd().split())
    grid = [rd().strip() for _ in range(H)]

    # --- initial statistics --------------------------------------------------
    row_size = [W] * H                 # number of remaining cookies in the row
    col_size = [H] * W                 # number of remaining cookies in the column

    row_cnt  = [[0] * 26 for _ in range(H)]   # colour frequency in each row
    col_cnt  = [[0] * 26 for _ in range(W)]   # colour frequency in each column

    for i in range(H):
        for j, ch in enumerate(grid[i]):
            c = ord(ch) - 97
            row_cnt[i][c] += 1
            col_cnt[j][c] += 1

    row_kind = [sum(1 for x in rc if x) for rc in row_cnt]   # distinct colours per row
    col_kind = [sum(1 for x in cc if x) for cc in col_cnt]   # distinct colours per col

    # alive[r][c] == 1  … cookie still present
    alive = [bytearray(b'\x01' * W) for _ in range(H)]

    # --- iterative deletion --------------------------------------------------
    while True:
        row_mark = [i for i in range(H) if row_size[i] >= 2 and row_kind[i] == 1]
        col_mark = [j for j in range(W) if col_size[j] >= 2 and col_kind[j] == 1]

        if not row_mark and not col_mark:
            break                                            # stable state reached

        # 1. remove all cookies in the marked rows
        for i in row_mark:
            for j in range(W):
                if alive[i][j]:
                    alive[i][j] = 0
                    c = ord(grid[i][j]) - 97

                    # update row information
                    row_cnt[i][c] -= 1
                    if row_cnt[i][c] == 0:
                        row_kind[i] -= 1
                    row_size[i] -= 1

                    # update column information
                    col_cnt[j][c] -= 1
                    if col_cnt[j][c] == 0:
                        col_kind[j] -= 1
                    col_size[j] -= 1

        # 2. remove all cookies in the marked columns (that are still alive)
        for j in col_mark:
            for i in range(H):
                if alive[i][j]:
                    alive[i][j] = 0
                    c = ord(grid[i][j]) - 97

                    # update column information
                    col_cnt[j][c] -= 1
                    if col_cnt[j][c] == 0:
                        col_kind[j] -= 1
                    col_size[j] -= 1

                    # update row information
                    row_cnt[i][c] -= 1
                    if row_cnt[i][c] == 0:
                        row_kind[i] -= 1
                    row_size[i] -= 1

    # -------------------------------------------------------------------------
    print(sum(row_size))


if __name__ == "__main__":
    main()