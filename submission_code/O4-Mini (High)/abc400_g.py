import sys
import threading
def main():
    import sys
    data = sys.stdin
    readline = data.readline
    INF = 10**30

    T = int(readline().strip())
    # Pre-define the possible (d1,d2) pairs for each parity cp
    # cp is a pair (p0,p1)
    pairs_by_cp = {
        (1,1): [(0,1),(1,0)],
        (1,0): [(0,2),(2,0)],
        (0,1): [(1,2),(2,1)],
    }

    out = []
    for _ in range(T):
        line = readline().split()
        # skip empty lines
        while not line:
            line = readline().split()
        N = int(line[0]); K = int(line[1])
        vals = [None]*N
        for i in range(N):
            x,y,z = map(int, readline().split())
            vals[i] = (x,y,z)
        # Build list of (A_i, i, D_i)
        arr = []
        arr_append = arr.append
        for i in range(N):
            x,y,z = vals[i]
            # find max and its dimension (tie-break to smallest index)
            if x >= y and x >= z:
                A = x; D = 0
            elif y >= x and y >= z:
                A = y; D = 1
            else:
                A = z; D = 2
            arr_append((A, i, D))
        # sort by A descending
        arr.sort(key=lambda x: x[0], reverse=True)
        M = 2*K
        # select top M
        S0 = arr[:M]
        sum_B = 0
        # parity counts
        cnt0 = cnt1 = cnt2 = 0
        for A,i,D in S0:
            sum_B += A
            if D == 0:
                cnt0 += 1
            elif D == 1:
                cnt1 += 1
            else:
                cnt2 += 1
        p0 = cnt0 & 1
        p1 = cnt1 & 1
        # if parity ok, no penalty
        if p0 == 0 and p1 == 0:
            out.append(str(sum_B))
            continue
        cp = (p0, p1)
        # get candidate pairs (d1,d2) for parity change
        pairs = pairs_by_cp[cp]

        # Compute P_A: penalty by internal reassign
        P_A = INF
        # For each selected rec
        for A,i,D in S0:
            # Only if this D matches a d1 in our pairs
            # For speed, check both pairs
            # Unpack v_i
            vi = vals[i]
            # iterate pairs
            # we know pairs small (size 2)
            for (d1,d2) in pairs:
                if D == d1:
                    # penalty = v_i[d1] - v_i[d2]
                    pen = vi[d1] - vi[d2]
                    if pen < P_A:
                        P_A = pen
        # Compute P_B: penalty by swapping one selected with one outside
        # First build M_unsel[d] = max v_j[d] over j not in S0
        if M < N:
            # there are unsel items
            M_unsel0 = -1  # since values non-negative
            M_unsel1 = -1
            M_unsel2 = -1
            for A,j,D in arr[M:]:
                vj = vals[j]
                # vj dims are >= 0
                x0,x1,x2 = vj
                if x0 > M_unsel0: M_unsel0 = x0
                if x1 > M_unsel1: M_unsel1 = x1
                if x2 > M_unsel2: M_unsel2 = x2
            # Now M_unsel is always >= 0
            P_B = INF
            # For each selected rec, consider removal from d1 and addition of best unsel in d2
            for A,i,D in S0:
                vi = vals[i]
                d1 = D
                # try both candidate pairs
                for (dd1,dd2) in pairs:
                    if d1 == dd1:
                        # penalty = v_i[d1] - M_unsel[dd2]
                        if dd2 == 0:
                            miss = M_unsel0
                        elif dd2 == 1:
                            miss = M_unsel1
                        else:
                            miss = M_unsel2
                        # if miss >= 0 always when unsel non-empty
                        pen = vi[d1] - miss
                        if pen < P_B:
                            P_B = pen
        else:
            # no unsel items
            P_B = INF

        P = P_A if P_A < P_B else P_B
        # P should be finite
        # subtract penalty
        ans = sum_B - P
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()