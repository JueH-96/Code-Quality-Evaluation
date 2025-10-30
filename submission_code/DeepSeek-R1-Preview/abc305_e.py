import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0

    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        adj[a].append(b)
        adj[b].append(a)

    max_stamina = [-1] * (N + 1)
    heap = []

    # Read guards and initialize heap
    for _ in range(K):
        p = int(input[idx])
        idx += 1
        h = int(input[idx])
        idx += 1
        max_stamina[p] = h
        heapq.heappush(heap, (-h, p))

    # Process the heap
    while heap:
        neg_stam, u = heapq.heappop(heap)
        current_stam = -neg_stam

        if current_stam < max_stamina[u]:
            continue

        for v in adj[u]:
            new_stam = current_stam - 1
            if new_stam > max_stamina[v]:
                max_stamina[v] = new_stam
                heapq.heappush(heap, (-new_stam, v))

    # Collect results
    result = []
    for v in range(1, N + 1):
        if max_stamina[v] >= 0:
            result.append(str(v))

    # Output
    print(len(result))
    print(' '.join(result))

if __name__ == '__main__':
    main()