import sys
import threading
import heapq

def main():
    import sys

    input = sys.stdin.readline
    N, M, K = map(int, input().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)

    # best_stamina[v] = the maximum remaining stamina when we reach v
    best_stamina = [-1] * (N+1)
    # max-heap of (stamina, node), store stamina as negative for heapq
    heap = []

    for _ in range(K):
        p, h = map(int, input().split())
        if best_stamina[p] < h:
            best_stamina[p] = h
            heapq.heappush(heap, (-h, p))

    while heap:
        neg_st, v = heapq.heappop(heap)
        st = -neg_st
        # If we've already found a better way to v, skip
        if st < best_stamina[v]:
            continue
        # Try to go to neighbors with st-1
        if st == 0:
            continue
        for w in adj[v]:
            nst = st - 1
            if best_stamina[w] < nst:
                best_stamina[w] = nst
                heapq.heappush(heap, (-nst, w))

    # Collect all vertices with best_stamina >= 0 (i.e., visited at all)
    res = [v for v in range(1, N+1) if best_stamina[v] >= 0]
    res.sort()
    print(len(res))
    if res:
        print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()