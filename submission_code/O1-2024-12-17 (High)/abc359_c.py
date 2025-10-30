def main():
    import sys
    data = sys.stdin.read().strip().split()
    Sx, Sy = map(int, data[:2])
    Tx, Ty = map(int, data[2:])

    # 1) The cost to cross horizontal tile boundaries is simply |Ty - Sy|.
    #    Whenever we move vertically at least one unit, we can freely choose
    #    which row(s) to do our horizontal moves in, thereby avoiding paying
    #    for vertical tile boundaries.  In other words, if |Ty - Sy| > 0,
    #    the minimum toll is just that vertical count.

    vertical_moves = abs(Ty - Sy)
    if vertical_moves > 0:
        print(vertical_moves)
        return

    # 2) If Sy == Ty, then we never change rows (no vertical moves).
    #    Hence we are stuck in one single row whose parity is Sy % 2.
    #    In that row, a vertical boundary "x = n" costs 1 exactly if
    #    n has the same parity as the row.  Otherwise it costs 0.
    #
    #    We only need to look at integer boundaries strictly between Sx and Tx.
    #    The number of such integers is (R - L - 1).  Among those, exactly about
    #    half have the same parity as Sy.  We do a small parity count.
    
    row_parity = Sy % 2
    L = min(Sx, Tx)
    R = max(Sx, Tx)
    count_in_between = R - L - 1  # how many integers strictly lie between L and R?

    if count_in_between <= 0:
        # No integer boundary in between, so no toll to pay
        print(0)
        return

    # Start boundary = L+1
    # We want to count how many of these count_in_between consecutive integers
    # match row_parity.

    # If (L+1) % 2 == row_parity, that first integer is "paying," so among
    # count_in_between consecutive numbers, ((count_in_between + 1) // 2) of them
    # match the same parity as the first; otherwise (count_in_between // 2).

    start_parity = (L + 1) % 2
    if start_parity == row_parity:
        # The first in-between integer has the same parity as the row,
        # so half (rounded up) of them share that parity
        toll = (count_in_between + 1) // 2
    else:
        # Otherwise half (rounded down) of them match the row parity
        toll = count_in_between // 2

    print(toll)

# Do not forget to call main()!
if __name__ == "__main__":
    main()