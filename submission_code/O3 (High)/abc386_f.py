import sys

INF_NEG = -10 ** 9  # sentinel for an unreachable state


def within_k_edit(s: str, t: str, k: int) -> bool:
    """
    Returns True  ⇔  the edit distance between s and t is ≤ k
    Implemented with Myers' O((N+M) + k²) algorithm (k ≤ 20).
    """
    n, m = len(s), len(t)

    # length difference is a lower bound
    if abs(n - m) > k:
        return False

    # constants for small arrays that hold diagonals
    OFFSET = k + 1                       # shift so that diag k=0 is at index OFFSET
    SIZE = 2 * k + 3                     # a little padding on both sides

    V_prev = [INF_NEG] * SIZE

    # longest common prefix – snakes for D = 0
    x = 0
    while x < n and x < m and s[x] == t[x]:
        x += 1
    V_prev[OFFSET] = x
    if x >= n and x >= m:
        return True                      # already identical

    # iterate over the allowed number of edits
    for d in range(1, k + 1):
        V_cur = [INF_NEG] * SIZE
        # diagonals with parity d have step 2
        for diag in range(-d, d + 1, 2):
            idx = diag + OFFSET

            # choose how we arrived here (insertion or deletion gives farther x)
            if diag == -d:
                # must have come from diag+1  (insertion into s : y++ )
                x_start = V_prev[idx + 1]            # insertion
            elif diag == d:
                # must have come from diag-1  (deletion from s : x++ )
                prev = V_prev[idx - 1]
                x_start = prev + 1 if prev != INF_NEG else INF_NEG
            else:
                # choose the bigger of insertion vs deletion
                ins = V_prev[idx + 1]                # insertion
                del_ = V_prev[idx - 1]
                del_ = del_ + 1 if del_ != INF_NEG else INF_NEG
                x_start = del_ if del_ > ins else ins

            if x_start < 0:                          # unreachable diagonal
                continue

            y_start = x_start - diag

            # extend the snake (match identical chars)
            x_tmp, y_tmp = x_start, y_start
            while x_tmp < n and y_tmp < m and s[x_tmp] == t[y_tmp]:
                x_tmp += 1
                y_tmp += 1

            V_cur[idx] = x_tmp

            # reached the end of both strings?
            if x_tmp >= n and y_tmp >= m:
                return True

        V_prev = V_cur                               # next iteration uses the row we just built

    return False                                     # need more than k edits


def main() -> None:
    data = sys.stdin.read().strip().splitlines()
    k = int(data[0])
    s = data[1].rstrip('
')
    t = data[2].rstrip('
')

    print("Yes" if within_k_edit(s, t, k) else "No")


if __name__ == "__main__":
    main()