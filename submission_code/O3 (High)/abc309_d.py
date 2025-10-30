import sys
from collections import deque


def bfs(start: int, g: list) -> list:
    """
    Breadth First Search that returns the shortest distance
    from `start` to every other vertex in the graph `g`.
    """
    n = len(g) - 1                    # because we keep index 0 unused
    dist = [-1] * (n + 1)
    dist[start] = 0
    dq = deque([start])

    while dq:
        v = dq.popleft()
        d_next = dist[v] + 1
        for nxt in g[v]:
            if dist[nxt] == -1:
                dist[nxt] = d_next
                dq.append(nxt)
    return dist


def main() -> None:
    input_data = sys.stdin.readline
    N1, N2, M = map(int, input_data().split())
    N = N1 + N2                       # total number of vertices

    # build undirected graph
    g = [[] for _ in range(N + 1)]    # 1-based indexing
    for _ in range(M):
        a, b = map(int, input_data().split())
        g[a].append(b)
        g[b].append(a)

    # shortest distances inside the two original components
    dist_from_1   = bfs(1,  g)
    dist_from_end = bfs(N, g)

    # farthest reachable vertices inside each component
    max_in_A = max(dist_from_1[1: N1 + 1])
    max_in_B = max(dist_from_end[N1 + 1: N + 1])

    # best we can do after adding one bridging edge
    answer = max_in_A + 1 + max_in_B
    print(answer)


if __name__ == "__main__":
    main()