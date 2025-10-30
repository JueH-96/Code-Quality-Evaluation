import sys


def main() -> None:
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    cells = sorted(zip(X, A))                 # sort by position

    total_stones = 0                          # prefix stones counted so far
    ans = 0                                   # total minimal operations
    prev = 0                                  # last processed position (0 at start)

    for pos, cnt in cells:
        # -------- handle gap without stones: (prev+1) .. (pos-1) -------------
        gap = pos - prev - 1
        if gap > 0:
            # need at least one stone per cell in this gap already
            if total_stones < pos - 1:        # S  < i  at i = pos-1
                print(-1)
                return

            # sum of c_i  in the gap
            a = prev + 1                      # first index in gap
            b = pos - 1                       # last index in gap
            # gap * total_stones  -  sum_{i=a}^{b} i
            gap_sum = gap * total_stones - (a + b) * gap // 2
            ans += gap_sum

        # ---------------- process current cell -------------------------------
        total_stones += cnt
        if pos < N:                           # boundaries only up to N-1
            c = total_stones - pos
            if c < 0:                         # prefix shortage
                print(-1)
                return
            ans += c

        prev = pos

    # ---------------- process tail after the last stone (prev+1 .. N-1) -------
    tail = (N - 1) - prev
    if tail > 0:
        if total_stones < N - 1:              # need at least N-1 stones before cell N
            print(-1)
            return

        a = prev + 1
        b = N - 1
        tail_sum = tail * total_stones - (a + b) * tail // 2
        ans += tail_sum

    # -------------------- final feasibility checks ----------------------------
    if total_stones != N:                     # wrong total amount
        print(-1)
        return

    print(ans)


if __name__ == "__main__":
    main()