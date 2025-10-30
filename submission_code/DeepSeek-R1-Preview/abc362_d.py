import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N

    adj = [[] for _ in range(N)]
    for _ in range(M):
        U = int(data[idx])
        idx += 1
        V = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        u = U - 1
        v = V - 1
        cost_uv = B + A[v]
        cost_vu = B + A[u]
        adj[u].append((v, cost_uv))
        adj[v].append((u, cost_vu))

    INF = float('inf')
    distance = [INF] * N
    distance[0] = A[0]
    heap = []
    heapq.heappush(heap, (distance[0], 0))

    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > distance[u]:
            continue
        for v, cost in adj[u]:
            if distance[v] > current_dist + cost:
                distance[v] = current_dist + cost
                heapq.heappush(heap, (distance[v], v))

    output = []
    for i in range(1, N):
        output.append(str(distance[i]))
    print(' '.join(output))

if __name__ == "__main__":
    main()