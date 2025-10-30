import sys
import threading

def main():
    import sys
    from heapq import heappush, heappop
    input = sys.stdin.readline

    N, M, K = map(int, input().split())
    # build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)

    # rem[v] = the maximum remaining stamina when reaching v (or -1 if unreachable)
    rem = [-1] * (N+1)
    heap = []  # max-heap by storing (-stamina, vertex)

    # initialize with the K guards
    for _ in range(K):
        p, h = map(int, input().split())
        rem[p] = h
        heappush(heap, (-h, p))

    # multi‑source BFS by descending remaining stamina
    while heap:
        neg_s, v = heappop(heap)
        s = -neg_s
        # if this entry is stale or no stamina left, skip
        if s != rem[v] or s == 0:
            continue
        # propagate to neighbors
        nxt = s - 1
        for u in adj[v]:
            if nxt > rem[u]:
                rem[u] = nxt
                heappush(heap, (-nxt, u))

    # collect all vertices with rem[v] >= 0
    ans = [str(v) for v in range(1, N+1) if rem[v] >= 0]
    print(len(ans))
    if ans:
        print(" ".join(ans))


if __name__ == "__main__":
    main()