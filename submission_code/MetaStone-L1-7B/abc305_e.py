import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1

    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        adj[a].append(b)
        adj[b].append(a)
    
    max_remaining = [-1] * (N + 1)
    heap = []
    for _ in range(K):
        p = int(data[idx])
        idx += 1
        h = int(data[idx])
        idx += 1
        max_remaining[p] = h
        heapq.heappush(heap, (-h, p))
    
    while heap:
        neg_current, u = heapq.heappop(heap)
        current_remaining = -neg_current
        if current_remaining < max_remaining[u]:
            continue
        for v in adj[u]:
            new_remaining = current_remaining - 1
            if new_remaining > max_remaining[v]:
                max_remaining[v] = new_remaining
                heapq.heappush(heap, (-new_remaining, v))
    
    result = [v for v in range(1, N+1) if max_remaining[v] >= 0]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()