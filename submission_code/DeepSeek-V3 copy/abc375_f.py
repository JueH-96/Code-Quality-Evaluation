import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    Q = int(data[idx+2])
    idx += 3
    edges = []
    for _ in range(M):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        edges.append((A, B, C))
        idx += 3
    closed = set()
    for _ in range(Q):
        type_q = int(data[idx])
        if type_q == 1:
            i = int(data[idx+1])
            closed.add(i-1)
            idx += 2
        else:
            x = int(data[idx+1])
            y = int(data[idx+2])
            idx += 3
            dist = [float('inf')] * (N+1)
            dist[x] = 0
            heap = []
            heapq.heappush(heap, (0, x))
            while heap:
                current_dist, u = heapq.heappop(heap)
                if u == y:
                    break
                if current_dist > dist[u]:
                    continue
                for idx_e, (A, B, C) in enumerate(edges):
                    if idx_e in closed:
                        continue
                    if A == u:
                        v = B
                        if dist[v] > dist[u] + C:
                            dist[v] = dist[u] + C
                            heapq.heappush(heap, (dist[v], v))
                    if B == u:
                        v = A
                        if dist[v] > dist[u] + C:
                            dist[v] = dist[u] + C
                            heapq.heappush(heap, (dist[v], v))
            if dist[y] == float('inf'):
                print(-1)
            else:
                print(dist[y])

if __name__ == "__main__":
    main()