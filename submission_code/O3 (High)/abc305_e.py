import sys
import heapq


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.readline

    # read N, M, K
    N, M, K = map(int, input_data().split())

    # build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input_data().split())
        adj[a].append(b)
        adj[b].append(a)

    # priority queue (max-heap by using negative value)
    pq = []
    best = [-1] * (N + 1)  # best[v] = maximum remaining stamina recorded for v

    # push guards into queue
    for _ in range(K):
        p, h = map(int, input_data().split())
        heapq.heappush(pq, (-h, p))

    # multi-source BFS with priority on larger remaining stamina
    while pq:
        neg_rem, v = heapq.heappop(pq)
        rem = -neg_rem
        if best[v] >= rem:
            continue               # we have already reached v with equal or better stamina

        best[v] = rem             # record current best stamina on v

        if rem == 0:              # no stamina left to move further
            continue

        # propagate to neighbours
        for nxt in adj[v]:
            nxt_rem = rem - 1
            if best[nxt] < nxt_rem:
                heapq.heappush(pq, (-nxt_rem, nxt))

    # collect guarded vertices
    guarded = [str(v) for v in range(1, N + 1) if best[v] >= 0]

    # output
    print(len(guarded))
    if guarded:
        print(' '.join(guarded))


if __name__ == "__main__":
    main()