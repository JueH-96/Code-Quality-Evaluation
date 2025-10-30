import sys
import heapq

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1

    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        a = int(input[idx]); idx += 1
        b = int(input[idx]); idx += 1
        adj[a].append(b)
        adj[b].append(a)

    max_reach = [-float('inf')] * (N + 1)
    heap = []

    for _ in range(K):
        p = int(input[idx]); idx += 1
        h = int(input[idx]); idx += 1
        if max_reach[p] < h:
            max_reach[p] = h
            heapq.heappush(heap, (-h, p))

    while heap:
        neg_r, u = heapq.heappop(heap)
        current_r = -neg_r
        if current_r < max_reach[u]:
            continue
        for v in adj[u]:
            new_r = current_r - 1
            if new_r > max_reach[v]:
                max_reach[v] = new_r
                heapq.heappush(heap, (-new_r, v))

    result = []
    for v in range(1, N + 1):
        if max_reach[v] >= 0:
            result.append(v)
    result.sort()

    print(len(result))
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()