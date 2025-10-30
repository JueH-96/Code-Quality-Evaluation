import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        weight_u_to_v = b + A[v - 1]
        weight_v_to_u = b + A[u - 1]
        adj[u].append((v, weight_u_to_v))
        adj[v].append((u, weight_v_to_u))

    INF = 1 << 60
    distance = [INF] * (N + 1)
    distance[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))

    while heap:
        d, u = heapq.heappop(heap)
        if d > distance[u]:
            continue
        for v, w in adj[u]:
            if distance[v] > d + w:
                distance[v] = d + w
                heapq.heappush(heap, (distance[v], v))

    a1 = A[0]
    res = []
    for i in range(2, N + 1):
        res.append(str(distance[i] + a1))
    print(' '.join(res))

if __name__ == '__main__':
    main()