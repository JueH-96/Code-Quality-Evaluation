import sys
import threading

def main():
    import sys
    import heapq

    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # For each station B, keep a list of (l, d, k, c, A) for infos that go A->B
    incoming = [[] for _ in range(N+1)]

    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        incoming[B].append((l, d, k, c, A))

    # f[s] = best (maximum) departure time from s that can still reach N.
    # Initialize f[N] = +INF, others = -1 (meaning unreachable).
    # We'll use an integer INF >> 1e18 (the max possible time).
    INF = 10**30
    f = [-1] * (N+1)
    f[N] = INF

    # Max-heap of (-f[s], s).  We use -f so the Python heapq (a min-heap) becomes a max-heap.
    heap = [(-INF, N)]
    seen = [False]*(N+1)

    while heap:
        neg_time, B = heapq.heappop(heap)
        curf = -neg_time
        # If this is stale or already finalized, skip
        if curf != f[B]:
            continue

        # Process each piece of info that arrives at B:
        for (l, d, k, c, A) in incoming[B]:
            # If f[B] is truly INF, we just take the very last train in that arith-prog
            if curf == INF:
                # last x = k-1
                tdep = l + d*(k-1)
            else:
                # otherwise we must have l + d*x + c <= curf
                limit = curf - c
                # if even the first train departs too late, skip
                if limit < l:
                    continue
                # largest x so that l + d*x <= limit
                x = (limit - l) // d
                if x < 0:
                    continue
                if x > k-1:
                    x = k-1
                tdep = l + d*x

            # relax f[A]
            if tdep > f[A]:
                f[A] = tdep
                heapq.heappush(heap, (-tdep, A))

    # Output
    out = []
    for s in range(1, N):
        if f[s] < 0:
            out.append("Unreachable")
        else:
            out.append(str(f[s]))
    sys.stdout.write("
".join(out) + "
")


if __name__ == "__main__":
    main()