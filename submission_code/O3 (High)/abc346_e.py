import sys
from collections import defaultdict


def main() -> None:
    # fast input
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    H = int(next(it))
    W = int(next(it))
    M = int(next(it))

    # store all operations
    ops = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(M)]

    # book-keeping of rows / columns whose final color is already fixed
    row_done = [False] * (H + 1)      # 1-indexed
    col_done = [False] * (W + 1)
    done_rows = 0
    done_cols = 0

    # resulting amount for every color
    color_cnt = defaultdict(int)

    # traverse operations in reverse order
    for t, a, x in reversed(ops):
        if t == 1:                               # row repaint
            if not row_done[a]:
                row_done[a] = True
                done_rows += 1
                cells = W - done_cols            # columns that are still undecided
                if cells:
                    color_cnt[x] += cells
        else:                                    # column repaint
            if not col_done[a]:
                col_done[a] = True
                done_cols += 1
                cells = H - done_rows            # rows that are still undecided
                if cells:
                    color_cnt[x] += cells

    # cells that never got touched by any operation keep color 0
    rest = (H - done_rows) * (W - done_cols)
    if rest:
        color_cnt[0] += rest

    # prepare output
    result = [(c, n) for c, n in color_cnt.items() if n > 0]
    result.sort()                                # ascending by color number

    print(len(result))
    for c, n in result:
        print(c, n)


if __name__ == "__main__":
    main()