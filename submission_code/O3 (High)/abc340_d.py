import sys
import heapq


def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    # adjacency list : for every stage keep (next_stage , time_cost)
    g = [[] for _ in range(N + 1)]

    for i in range(1, N):
        a, b, x = map(int, input().split())
        g[i].append((i + 1, a))  # clear with A_i → unlock i+1
        g[i].append((x, b))      # clear with B_i → unlock X_i

    INF = 10 ** 20
    dist = [INF] * (N + 1)
    dist[1] = 0

    pq = [(0, 1)]  # (current_time, stage)

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == N:                 # we already reached stage N with minimal time
            break
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    print(dist[N])


if __name__ == "__main__":
    main()