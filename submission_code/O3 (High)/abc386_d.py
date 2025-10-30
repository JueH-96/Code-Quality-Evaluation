import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())

    # row_id -> [lower_bound , upper_bound]
    # lower_bound : largest column that must be black in this row
    # upper_bound : largest column that is still allowed to be black
    row_bounds = {}

    for _ in range(M):
        x_str, y_str, colour = sys.stdin.readline().split()
        x = int(x_str)
        y = int(y_str)

        if x not in row_bounds:
            # initially no restriction: any value between 0 and N
            row_bounds[x] = [0, N]

        if colour == 'B':          # black => i_x >= y
            row_bounds[x][0] = max(row_bounds[x][0], y)
        else:                      # white => i_x < y  ⇒  i_x ≤ y-1
            row_bounds[x][1] = min(row_bounds[x][1], y - 1)

    # 1. every row must have a non-empty interval
    for low, up in row_bounds.values():
        if low > up:
            print("No")
            return

    # 2. choose the sequence i_1 … i_N so that it is non-increasing
    #    and respects every interval [low, up]
    current = N                          # i_0 (virtual) = N
    for r in sorted(row_bounds):         # rows in increasing order
        low, up = row_bounds[r]
        current = min(current, up)       # i_r cannot exceed previous value
        if current < low:                # interval empty → impossible
            print("No")
            return
        # otherwise fix i_r = current and continue

    # all constraints satisfied
    print("Yes")


if __name__ == "__main__":
    main()