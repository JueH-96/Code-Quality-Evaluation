import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N+1)]  # 1-based indexing
    roads = []
    for i in range(1, M+1):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        roads.append((A, B, C))
        adj[A].append((B, C, i))
        adj[B].append((A, C, i))
    
    closed = [False] * (M + 2)  # indices 1..M

    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            i = int(input[ptr])
            ptr += 1
            closed[i] = True
        else:
            x = int(input[ptr])
            ptr += 1
            y = int(input[ptr])
            ptr += 1
            # Run Dijkstra from x to y, avoiding closed roads
            INF = float('inf')
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            found = False
            while heap:
                d, u = heapq.heappop(heap)
                if u == y:
                    print(d)
                    found = True
                    break
                if d > dist[u]:
                    continue
                for v, c, idx in adj[u]:
                    if closed[idx]:
                        continue
                    if dist[v] > d + c:
                        dist[v] = d + c
                        heapq.heappush(heap, (dist[v], v))
            if not found:
                print(-1)

if __name__ == '__main__':
    main()