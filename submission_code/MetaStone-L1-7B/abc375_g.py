import sys
import heapq

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    edges = []
    adj = [[] for _ in range(N+1)]  # 1-based
    for _ in range(M):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        C = int(input[idx])
        idx += 1
        edges.append((A, B, C))
        adj[A].append((B, C))

    # Compute dist1 from 1 using Dijkstra
    dist1 = [float('inf')] * (N + 1)
    dist1[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist1[u]:
            continue
        for v, c in adj[u]:
            if dist1[v] > dist1[u] + c:
                dist1[v] = dist1[u] + c
                heapq.heappush(heap, (dist1[v], v))

    # Compute distN from N using reversed graph
    reversed_adj = [[] for _ in range(N+1)]
    for A, B, C in edges:
        reversed_adj[B].append((A, C))
    distN = [float('inf')] * (N + 1)
    distN[N] = 0
    heap = []
    heapq.heappush(heap, (0, N))
    while heap:
        d, u = heapq.heappop(heap)
        if d > distN[u]:
            continue
        for v, c in reversed_adj[u]:
            if distN[v] > distN[u] + c:
                distN[v] = distN[u] + c
                heapq.heappush(heap, (distN[v], v))

    # Build filtered_edges for each node
    filtered_edges = {u: [] for u in range(N+1)}
    for A, B, C in edges:
        if dist1[A] + C == dist1[B]:
            filtered_edges[A].append((B, C))

    # Process each edge
    output = []
    for A, B, C in edges:
        if dist1[A] + C != dist1[B] and dist1[B] + C != dist1[A]:
            output.append("No")
        else:
            if dist1[A] + C == dist1[B]:
                if len(filtered_edges[A]) > 1:
                    output.append("No")
                else:
                    output.append("Yes")
            else:
                output.append("Yes")

    print('
'.join(output))

if __name__ == '__main__':
    main()