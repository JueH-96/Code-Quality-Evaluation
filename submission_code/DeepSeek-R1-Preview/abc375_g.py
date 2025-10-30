import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    roads = []
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        roads.append((A, B, C))

    # Build original adjacency list
    adj_original = [[] for _ in range(N+1)]
    for A, B, C in roads:
        adj_original[A].append((B, C))
        adj_original[B].append((A, C))

    # Dijkstra from 1
    INF = float('inf')
    dist1 = [INF] * (N + 1)
    dist1[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist1[u]:
            continue
        for v, c in adj_original[u]:
            if dist1[v] > d + c:
                dist1[v] = d + c
                heapq.heappush(heap, (dist1[v], v))

    # Dijkstra from N
    distN = [INF] * (N + 1)
    distN[N] = 0
    heap = []
    heapq.heappush(heap, (0, N))
    while heap:
        d, u = heapq.heappop(heap)
        if d > distN[u]:
            continue
        for v, c in adj_original[u]:
            if distN[v] > d + c:
                distN[v] = d + c
                heapq.heappush(heap, (distN[v], v))

    D = dist1[N]

    # Build S
    S = set()
    adj_S = [[] for _ in range(N + 1)]
    for i in range(M):
        A, B, C = roads[i]
        if (dist1[A] + C + distN[B] == D) or (dist1[B] + C + distN[A] == D):
            S.add(i)
            adj_S[A].append((B, i))
            adj_S[B].append((A, i))

    # Tarjan's algorithm to find bridges iteratively
    bridges = set()
    disc = [0] * (N + 1)
    low = [0] * (N + 1)
    visited = [False] * (N + 1)
    time = 1

    for u in range(1, N + 1):
        if not visited[u]:
            stack = []
            parent = {}  # To track parent edge for each node
            stack.append((u, -1, False))
            while stack:
                node, edge_id, processed = stack.pop()
                if not processed:
                    if visited[node]:
                        continue
                    visited[node] = True
                    disc[node] = low[node] = time
                    time += 1
                    stack.append((node, edge_id, True))
                    # Push children
                    neighbors = adj_S[node]
                    # Reverse to process in correct order
                    neighbors = reversed(neighbors)
                    for v, e in neighbors:
                        if e == edge_id:
                            continue
                        if not visited[v]:
                            parent[v] = e
                            stack.append((v, e, False))
                        else:
                            # Back edge
                            if e != edge_id:
                                low[node] = min(low[node], disc[v])
                else:
                    for v, e in adj_S[node]:
                        if e == edge_id:
                            continue
                        if v in parent and parent[v] == e:
                            # It's a child, update low
                            low[node] = min(low[node], low[v])
                            if low[v] > disc[node]:
                                bridges.add(e)
                        else:
                            # Back edge
                            low[node] = min(low[node], disc[v])

    # Output results
    for i in range(M):
        if i in S:
            print("Yes" if i in bridges else "No")
        else:
            print("No")

if __name__ == '__main__':
    main()