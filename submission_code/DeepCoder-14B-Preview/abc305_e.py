import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(M):
        a = int(input[idx]); idx += 1
        b = int(input[idx]); idx += 1
        adj[a].append(b)
        adj[b].append(a)

    max_stamina = [-1] * (N + 1)
    heap = []

    # Read guards' positions and stamina
    for _ in range(K):
        p = int(input[idx]); idx += 1
        h = int(input[idx]); idx += 1
        max_stamina[p] = h
        heapq.heappush(heap, (-h, p))  # Using max-heap via min-heap with negative values

    # Process the heap
    while heap:
        neg_s, u = heapq.heappop(heap)
        current_s = -neg_s

        # If current_s is less than the recorded max stamina, skip processing
        if current_s < max_stamina[u]:
            continue

        # Explore neighbors
        for v in adj[u]:
            new_s = current_s - 1
            if new_s > max_stamina[v]:
                max_stamina[v] = new_s
                heapq.heappush(heap, (-new_s, v))

    # Collect guarded vertices
    guarded = [v for v in range(1, N + 1) if max_stamina[v] >= 0]
    guarded.sort()

    # Output the result
    print(len(guarded))
    if guarded:
        print(' '.join(map(str, guarded)))

if __name__ == '__main__':
    main()