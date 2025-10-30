def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # initial positions
    l = 1
    r = 2
    total_moves = 0

    for _ in range(Q):
        H, T = input().split()
        T = int(T)
        if H == 'L':
            S, F = l, r
        else:
            S, F = r, l

        # compute the two candidate lengths on the cycle
        cw_len  = (T - S) % N  # steps moving forward S-->S+1-->...-->T
        ccw_len = (S - T) % N  # steps moving backward S-->S-1-->...-->T

        # check if forbidden node F lies on the open-interval path in that direction
        # for cw: distance from S to F clockwise is dSF; if 1 <= dSF <= cw_len, cw is blocked.
        dSF = (F - S) % N
        cw_blocked = (1 <= dSF <= cw_len)

        # for ccw: distance from S to F going counterclockwise is dSC; if 1 <= dSC <= ccw_len, ccw is blocked.
        dSC = (S - F) % N
        ccw_blocked = (1 <= dSC <= ccw_len)

        # pick the minimum valid distance
        cand = []
        if not cw_blocked:
            cand.append(cw_len)
        if not ccw_blocked:
            cand.append(ccw_len)
        # at least one must be valid by problem guarantee
        move_cost = min(cand)

        total_moves += move_cost

        # update the moved hand
        if H == 'L':
            l = T
        else:
            r = T

    print(total_moves)

if __name__ == "__main__":
    main()