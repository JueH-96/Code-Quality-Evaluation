def solve():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)

    # A < B is guaranteed by the problem statement.
    # The 3x3 board looks like this:
    # 1 2 3
    # 4 5 6
    # 7 8 9
    #
    # The pairs that are adjacent horizontally are:
    # (1,2), (2,3),
    # (4,5), (5,6),
    # (7,8), (8,9).
    #
    # Thus, A and B are adjacent horizontally if:
    # 1) They differ by 1 (B == A + 1)
    # 2) A is not on the rightmost boundary (i.e., A != 3, 6, 9)

    if B == A + 1 and A not in (3, 6, 9):
        print("Yes")
    else:
        print("No")

# Call solve() to execute
solve()