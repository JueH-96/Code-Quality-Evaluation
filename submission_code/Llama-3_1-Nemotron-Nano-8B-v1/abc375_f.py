import sys
import heapq

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    adj = [[] for _ in range(N+1)]  # 1-based
    edges = []
    for i in range(1, M+1):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        adj[A].append((B, C, i))
        adj[B].append((A, C, i))
        edges.append((A, B, C, i))  # Not used but needed for order

    closed = [False] * (M + 2)  # 1-based edge indices

    for _ in range(Q):
        query = input[ptr]
        ptr +=1
        if query == '1':
            i = int(input[ptr]); ptr +=1
            closed[i] = True
        else:
            x = int(input[ptr]); ptr +=1
            y = int(input[ptr]); ptr +=1
            # Dijkstra's algorithm
            INF = float('inf')
            dist = [INF] * (N + 1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            visited = [False] * (N + 1)
            found = False
            while heap:
                d, u = heapq.heappop(heap)
                if u == y:
                    found = True
                    break
                if visited[u]:
                    continue
                visited[u] = True
                for v, c, idx in adj[u]:
                    if closed[idx]:
                        continue
                    if dist[v] > d + c:
                        dist[v] = d + c
                        heapq.heappush(heap, (dist[v], v))
            print(dist[y] if found and dist[y] != INF else -1)

if __name__ == '__main__':
    main()