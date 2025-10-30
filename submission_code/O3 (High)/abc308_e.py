import sys

def build_mex_table():
    """pre-compute mex for every triple of values in {0,1,2}"""
    table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if 0 not in (a, b, c):
                    m = 0
                elif 1 not in (a, b, c):
                    m = 1
                elif 2 not in (a, b, c):
                    m = 2
                else:
                    m = 3
                table[a][b][c] = m
    return table


def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))

    A = [int(next(it)) for _ in range(n)]
    S = next(it)

    # count of X’s to the right of current index for each value {0,1,2}
    right_cnt = [0, 0, 0]
    for val, ch in zip(A, S):
        if ch == 'X':
            right_cnt[val] += 1

    left_cnt = [0, 0, 0]                      # count of M’s to the left
    mex_table = build_mex_table()
    ans = 0

    for val, ch in zip(A, S):
        if ch == 'X':
            # this X will no longer be on the right for future positions
            right_cnt[val] -= 1
            continue

        if ch == 'E':
            # add contribution of every (M, current E, X) combination
            for vM in range(3):
                lm = left_cnt[vM]
                if lm == 0:
                    continue
                for vX in range(3):
                    rx = right_cnt[vX]
                    if rx == 0:
                        continue
                    ans += lm * rx * mex_table[vM][val][vX]

        elif ch == 'M':
            # current position becomes available for future tuples
            left_cnt[val] += 1

    print(ans)


if __name__ == "__main__":
    main()