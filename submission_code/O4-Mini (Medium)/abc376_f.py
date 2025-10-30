def main():
    import sys
    input = sys.stdin.readline
    INF = 10**18

    N, Q = map(int, input().split())
    instr = [input().split() for _ in range(Q)]

    # dp[p] = min cost ending with the "fixed" hand at f_old
    # and the other hand at position p (p != f_old).
    # We also track which hand was "fixed" last: old_hand in {'L','R'}.
    # Initially: left hand at 1 is "fixed", right hand at 2 is "other".
    old_hand = 'L'
    f_old = 1
    dp = [INF] * (N+1)
    dp[2] = 0        # right hand starts at 2

    for H_new, sT in instr:
        T = int(sT)
        dp_new = [INF] * (N+1)

        # We will move whichever hand H_new to T;
        # the other hand will end at some q != T.
        for p in range(1, N+1):
            cost0 = dp[p]
            if cost0 >= INF: 
                continue

            # Unpack current (L=u, R=v)
            if old_hand == 'L':
                u, v = f_old, p
            else:
                u, v = p, f_old

            # hpos = old position of the hand we're about to move
            # op   = old position of the other hand
            if H_new == 'L':
                hpos, op = u, v
            else:
                hpos, op = v, u

            # distance CW and CCW around the cycle
            d_cw  = (T - hpos) % N
            d_ccw = (hpos - T) % N

            # consider whichever direction is shorter; if tied, do both
            for dir_flag, d in ((1, d_cw), (-1, d_ccw)) if d_cw == d_ccw else ((1, d_cw),) if d_cw < d_ccw else ((-1, d_ccw),):
                # dir_flag = +1 means we walk CW, -1 means CCW
                # nodes on the path are hpos + k*dir_flag, k=1..d
                # check if op lies on that arc
                if dir_flag == 1:
                    diff = (op - hpos) % N
                else:
                    diff = (hpos - op) % N

                if 1 <= diff <= d:
                    # blocked: must shift the other hand 1 step off the path
                    # that off-path neighbor is exactly `op - dir_flag`
                    if dir_flag == 1:
                        # we were going CW, so off-path is stepping CCW
                        new_op = op-1 if op>1 else N
                    else:
                        # going CCW, off-path is stepping CW
                        new_op = op+1 if op<N else 1
                    if new_op != T:
                        c = cost0 + d + 1
                        if c < dp_new[new_op]:
                            dp_new[new_op] = c
                else:
                    # not blocked: we can move the instructed hand straight in d steps
                    # other remains at op
                    if op != T:
                        c = cost0 + d
                        if c < dp_new[op]:
                            dp_new[op] = c

        # swap in the new DP, record the newly fixed hand
        dp = dp_new
        old_hand = H_new
        f_old = T

    # at the end dp[p] is the cost when the last-instructed hand sits at f_old=T_Q
    # and the other is at p!=T_Q.  answer = min over p
    ans = min(dp[p] for p in range(1, N+1) if p != f_old)
    print(ans)


if __name__ == "__main__":
    main()