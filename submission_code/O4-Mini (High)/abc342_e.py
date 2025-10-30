#!/usr/bin/env python3
import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # b2info[v] = list of infos i with B_i = v.
    # store each as tuple (A_i, l_i, d_i, k_i, c_i)
    b2info = [[] for _ in range(N+1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        b2info[B].append((A, l, d, k, c))
    # dp[u] = the latest departure time from u that can still reach N.
    # initialize all unreachable = -1, dp[N] = INF
    INF = 10**30
    dp = [-1] * (N+1)
    dp[N] = INF
    # inqueue flag to avoid duplicates
    inqueue = [False] * (N+1)
    dq = deque()
    dq.append(N)
    inqueue[N] = True

    while dq:
        v = dq.popleft()
        inqueue[v] = False
        dpv = dp[v]
        # if dpv < 0: unreachable, skip
        if dpv < 0:
            continue
        # process all train‐infos that arrive at v
        for (A, l, d, k, c) in b2info[v]:
            # compute best departure t_best from A via this info
            # we need t + c <= dp[v], so t <= dpv - c
            if dpv == INF:
                # from N we treat as infinite arrival window => can take last train
                j = k - 1
            else:
                Tlim = dpv - c
                if Tlim < l:
                    continue
                # floor((Tlim - l)/d)
                j = (Tlim - l) // d
                if j < 0:
                    continue
                if j > k - 1:
                    j = k - 1
            t_best = l + d * j
            # relax dp[A]
            if t_best > dp[A]:
                dp[A] = t_best
                if not inqueue[A]:
                    inqueue[A] = True
                    dq.append(A)

    # output
    out = []
    for u in range(1, N):
        if dp[u] < 0:
            out.append("Unreachable")
        else:
            out.append(str(dp[u]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()