import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    C = list(map(int, data[2:]))

    INF = 10**30
    # dp[t][v]: min cost up to current position,
    # with t equal‐adjacencies seen so far (0 or 1),
    # and current bit = v (0 or 1).
    dp = [[INF, INF], [INF, INF]]
    # Initialize at position 0 (1 in problem)
    s0 = int(S[0])
    for v in (0, 1):
        dp[0][v] = C[0] if v != s0 else 0

    # Process positions 1..N-1
    for i in range(1, N):
        si = int(S[i])
        ci = C[i]
        dp2 = [[INF, INF], [INF, INF]]
        for t_prev in (0, 1):
            row = dp[t_prev]
            # if both row entries are INF, skip
            if row[0] >= INF and row[1] >= INF:
                continue
            for v_prev in (0, 1):
                cost_prev = row[v_prev]
                if cost_prev >= INF:
                    continue
                # choose v_cur
                # two possibilities: 0 or 1
                # compute cost to flip or not
                # and new t count
                # v = 0
                # v = 1
                # write inline
                # v_cur = 0
                for v_cur in (0, 1):
                    t_new = t_prev + (1 if v_cur == v_prev else 0)
                    if t_new <= 1:
                        add = 0 if v_cur == si else ci
                        cnew = cost_prev + add
                        if cnew < dp2[t_new][v_cur]:
                            dp2[t_new][v_cur] = cnew
        dp = dp2

    # We need exactly 1 equal adjacency
    ans = min(dp[1][0], dp[1][1])
    print(ans)


if __name__ == "__main__":
    main()