import sys
import heapq

def dijkstra(start, N, adj):
    INF = 1 << 60
    dist = [INF] * (N + 1)
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, c in adj[u]:
            if dist[v] > d + c:
                dist[v] = d + c
                heapq.heappush(heap, (dist[v], v))
    return dist

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    adj = [[] for _ in range(N + 1)]
    roads = []
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        adj[A].append((B, C))
        adj[B].append((A, C))
        roads.append((A, B, C))
    d1 = dijkstra(1, N, adj)
    dn = dijkstra(N, N, adj)
    D = d1[N]
    for A, B, C in roads:
        cond1 = (d1[A] + C + dn[B] == D)
        cond2 = (d1[B] + C + dn[A] == D)
        if not (cond1 or cond2):
            print("No")
        else:
            if (d1[A] + C == d1[B] and dn[B] + C == dn[A]) or (d1[B] + C == d1[A] and dn[A] + C == dn[B]):
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    main()