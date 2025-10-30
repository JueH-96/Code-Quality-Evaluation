import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    stages = [ (0, 0, 0) ] * (N + 1)  # 1-based indexing for stages 1 to N-1
    
    for i in range(1, N):
        A = int(data[idx])
        B = int(data[idx + 1])
        X = int(data[idx + 2])
        stages[i] = (A, B, X)
        idx += 3
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if u == N:
            print(d)
            return
        if d > dist[u]:
            continue
        if u >= N:
            continue
        A, B, X = stages[u]
        # Option A: move to u+1
        v = u + 1
        if v <= N:
            new_d = d + A
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))
        # Option B: move to X
        v = X
        if v <= N:
            new_d = d + B
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))
    
    # The problem guarantees that stage N is reachable, so this line is just in case
    print(-1)

if __name__ == '__main__':
    main()