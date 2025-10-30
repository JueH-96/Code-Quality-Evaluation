def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))

    # Special case: M = 2.
    # In this case, the only values possible are 0 and 1. Because A and B are "good",
    # they must alternate 0,1,... or 1,0,... . Any single +1 or -1 step would produce
    # two adjacent equal values unless A == B already (in which case no steps are needed).
    if M == 2:
        if A == B:
            print(0)
        else:
            print(-1)
        return

    # For M >= 3, one can show (and is also suggested by the given samples) that
    # it is always possible to re-order increments/decrements so that adjacency
    # is never broken, and the minimal number of operations equals the sum of
    # the individual "circular distances" from A_i to B_i.
    #
    # The circular distance from x to y (mod M) by single ±1 steps is:
    #   d = min( (y - x) mod M, (x - y) mod M ).
    #
    # We simply sum this over i = 1..N.

    total_ops = 0
    for i in range(N):
        diff = (B[i] - A[i]) % M
        total_ops += min(diff, M - diff)

    print(total_ops)

# Don't forget to call main()!
if __name__ == "__main__":
    main()