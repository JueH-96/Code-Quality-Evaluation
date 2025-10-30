import sys


def main() -> None:
    input = sys.stdin.readline
    sys.setrecursionlimit(1 << 25)

    N, W = map(int, input().split())

    # store blocks in each column: (Y, index)
    cols = [[] for _ in range(W)]
    for idx in range(N):
        x, y = map(int, input().split())
        cols[x - 1].append((y, idx))

    # minimum number of blocks in any column (row-clear can occur at most this many times)
    min_size = min(len(c) for c in cols)

    # rank (1-indexed from the bottom) of every block inside its column
    rank_of = [0] * N

    # when min_size == 0, no row will ever be cleared
    if min_size:
        max_gap = [0] * min_size  # max gap for every rank (0 … min_size-1)

    # sort every column, compute ranks and gaps
    for col in cols:
        col.sort(key=lambda t: t[0])           # sort by Y ascending
        for r, (y, idx) in enumerate(col):
            rank_of[idx] = r + 1               # ranks start from 1

        if not min_size:
            continue

        ys = [y for y, _ in col]               # only Y coordinates
        # first gap is the height of the first block itself
        gap = ys[0]
        if gap > max_gap[0]:
            max_gap[0] = gap
        # remaining gaps
        for r in range(1, min_size):
            gap = ys[r] - ys[r - 1]
            if gap > max_gap[r]:
                max_gap[r] = gap

    # cumulative times t_r (1-indexed) at which the r-th row clear happens
    times = []
    if min_size:
        acc = 0
        for g in max_gap:
            acc += g
            times.append(acc)       # times[r-1] == t_r

    # removal time for every block (INF if never removed)
    INF = 10 ** 30
    remove_time = [INF] * N
    if min_size:
        for idx in range(N):
            r = rank_of[idx]
            if r <= min_size:
                remove_time[idx] = times[r - 1]

    # answer queries
    Q = int(input())
    out_lines = []
    for _ in range(Q):
        T, A = map(int, input().split())
        if T >= remove_time[A - 1]:
            out_lines.append("No")
        else:
            out_lines.append("Yes")

    print("
".join(out_lines))


if __name__ == "__main__":
    main()