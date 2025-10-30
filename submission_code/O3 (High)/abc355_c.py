import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, T = data[0], data[1]
    A = data[2:2 + T]

    row_cnt = [0] * N
    col_cnt = [0] * N
    diag_main = 0          # top-left  -> bottom-right
    diag_anti = 0          # top-right -> bottom-left

    for turn, x in enumerate(A, 1):
        x -= 1                 # to 0-based
        r = x // N             # row index 0 … N-1
        c = x %  N             # col index 0 … N-1

        # update row
        row_cnt[r] += 1
        if row_cnt[r] == N:
            print(turn)
            return

        # update column
        col_cnt[c] += 1
        if col_cnt[c] == N:
            print(turn)
            return

        # update main diagonal
        if r == c:
            diag_main += 1
            if diag_main == N:
                print(turn)
                return

        # update anti-diagonal
        if r + c == N - 1:
            diag_anti += 1
            if diag_anti == N:
                print(turn)
                return

    # no bingo
    print(-1)

if __name__ == "__main__":
    main()