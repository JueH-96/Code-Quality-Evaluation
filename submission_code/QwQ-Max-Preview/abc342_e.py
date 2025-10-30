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
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        l = int(data[idx])
        idx += 1
        d = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        c = int(data[idx])
        idx += 1
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        adj[B].append((l, d, k, c, A))
    
    INF = 10**30
    latest = [-float('inf')] * (N + 1)
    latest[N] = INF
    heap = [(-INF, N)]
    
    while heap:
        current_t_neg, u = heapq.heappop(heap)
        current_t = -current_t_neg
        if current_t < latest[u]:
            continue
        for edge in adj[u]:
            l, d, k, c, A = edge
            target = current_t - c
            if target < l:
                continue
            max_m = (target - l) // d
            if max_m < 0:
                continue
            max_m = min(max_m, k - 1)
            t = l + max_m * d
            if t > latest[A]:
                latest[A] = t
                heapq.heappush(heap, (-t, A))
    
    for i in range(1, N):
        if latest[i] == -float('inf'):
            print("Unreachable")
        else:
            print(latest[i])

if __name__ == "__main__":
    main()