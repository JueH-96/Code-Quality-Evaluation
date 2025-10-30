def main():
    import sys
    import heapq

    data = sys.stdin.read().strip().split()
    idx = 0

    N = int(data[idx]); idx+=1
    M = int(data[idx]); idx+=1
    K = int(data[idx]); idx+=1

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx]); idx+=1
        b = int(data[idx]); idx+=1
        adj[a].append(b)
        adj[b].append(a)

    # Read guards information
    guards = []
    for _ in range(K):
        p = int(data[idx]); idx+=1
        h = int(data[idx]); idx+=1
        guards.append((p,h))
    
    # For each vertex, store the maximum "coverage" (distance left) found so far.
    coverage = [-1]*(N+1)

    # Max-heap (using negative values to simulate max-heap in Python)
    hq = []
    for p, h in guards:
        heapq.heappush(hq, (-h, p))  # store as (-h, vertex)

    # Multi-source BFS with priority queue
    while hq:
        dist_left, v = heapq.heappop(hq)
        dist_left = -dist_left  # convert back to positive

        # If we've already found a coverage >= dist_left for this node, skip
        if coverage[v] >= dist_left:
            continue

        # Update coverage
        coverage[v] = dist_left

        # If we can still move, push neighbors with one less coverage
        if dist_left > 0:
            for w in adj[v]:
                if coverage[w] < dist_left - 1:
                    heapq.heappush(hq, (-(dist_left - 1), w))

    # Gather and sort guarded vertices
    guarded_vertices = [i for i in range(1, N+1) if coverage[i] >= 0]
    guarded_vertices.sort()

    # Output the result
    print(len(guarded_vertices))
    print(*guarded_vertices)

# Call main() at the end
if __name__ == "__main__":
    main()