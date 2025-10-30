import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input().strip())
    grid = []
    row_cnt = [0] * N
    col_cnt = [0] * N

    # first pass : read grid and count 'o' per row/column
    for i in range(N):
        s = input().strip()
        grid.append(s)
        for j, ch in enumerate(s):
            if ch == 'o':
                row_cnt[i] += 1
                col_cnt[j] += 1

    # second pass : for every 'o' cell use it as the pivot
    # contribution = (others in its row) * (others in its column)
    ans = 0
    for i, s in enumerate(grid):
        r_other = row_cnt[i] - 1           # cells with 'o' in the same row except itself
        if r_other == 0:                   # quick skip if no mate in row
            continue
        for j, ch in enumerate(s):
            if ch == 'o':
                c_other = col_cnt[j] - 1   # cells with 'o' in the same column except itself
                if c_other:
                    ans += r_other * c_other

    print(ans)

if __name__ == "__main__":
    main()