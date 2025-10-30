import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    infos = []
    # adjacency for reachability: reverse edges B->A
    revg = [[] for _ in range(N+1)]
    # for relaxation: for each B, list of infos indices
    byB = [[] for _ in range(N+1)]

    for i in range(M):
        l,d,k,c,A,B = map(int, input().split())
        infos.append((l, d, k, c, A, B))
        revg[B].append(A)
        byB[B].append(i)

    # 1) BFS from N on revg to mark reachable-to-N
    from collections import deque
    reachable = [False]*(N+1)
    dq = deque([N])
    reachable[N] = True
    while dq:
        u = dq.popleft()
        for v in revg[u]:
            if not reachable[v]:
                reachable[v] = True
                dq.append(v)

    # 2) dp[u] = best departure time from u, init -inf, dp[N]=INF
    INF = 10**30
    dp = [-10**30] * (N+1)
    dp[N] = INF

    # 3) max-heap of (dp, u)
    import heapq
    pq = []
    heapq.heappush(pq, (-dp[N], N))

    while pq:
        nd, u = heapq.heappop(pq)
        dpu = -nd
        if dpu != dp[u]:
            continue
        # relax all infos that arrive at u
        for idx in byB[u]:
            l, d, k, c, A, B = infos[idx]
            # skip if A not reachable
            if not reachable[A]:
                continue
            # can we depart from A via this info?
            # Need t + c <= dp[B] == dp[u]
            cap = dpu - c
            if cap < l:
                continue
            # find largest j in [0,k-1] with l + d*j <= cap
            j = (cap - l) // d
            if j >= k:
                j = k - 1
            t = l + d*j
            if t > dp[A]:
                dp[A] = t
                heapq.heappush(pq, (-t, A))

    # 4) Output
    out = []
    for s in range(1, N):
        if not reachable[s]:
            out.append("Unreachable")
        else:
            # dp[s] must be finite
            out.append(str(dp[s]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()