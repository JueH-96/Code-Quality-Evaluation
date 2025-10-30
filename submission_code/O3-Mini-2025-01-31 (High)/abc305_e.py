def main():
    import sys
    import heapq
    input = sys.stdin.readline

    # Read the number of vertices, edges, and guards
    N, M, K = map(int, input().split())
    graph = [[] for _ in range(N + 1)]
    
    # Build the graph using an adjacency list.
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
        
    # best[v] will store the maximum remaining stamina with which a guard can reach vertex v.
    # If best[v] is >= 0, then vertex v is guarded by some guard.
    best = [-1] * (N + 1)
    
    # We'll perform a multi-source "BFS"-like traversal.
    # Each guard starts as a source at vertex p with stamina h.
    # Use a max-heap (implemented via negating values in heapq) so that we propagate from higher remaining stamina first.
    heap = []
    for _ in range(K):
        p, h = map(int, input().split())
        best[p] = h
        heapq.heappush(heap, (-h, p))
    
    # Propagate the guard's "coverage" across the graph.
    while heap:
        neg_energy, cur = heapq.heappop(heap)
        cur_energy = -neg_energy
        
        # Skip if this entry is outdated.
        if best[cur] != cur_energy:
            continue
        
        # If no stamina is left to propagate, skip the neighbors.
        if cur_energy == 0:
            continue
        
        # Try to propagate to each neighbor.
        for neighbor in graph[cur]:
            next_energy = cur_energy - 1
            # Only update if this new energy is higher than the best we've seen for neighbor.
            if next_energy > best[neighbor]:
                best[neighbor] = next_energy
                heapq.heappush(heap, (-next_energy, neighbor))
    
    # Collect all the vertices that are guarded (i.e. with best[v] >= 0).
    guarded = [v for v in range(1, N + 1) if best[v] >= 0]
    guarded.sort()
    
    # Output the results:
    # First, the number of guarded vertices, then the list of guarded vertices in ascending order.
    sys.stdout.write(str(len(guarded)) + "
")
    sys.stdout.write(" ".join(map(str, guarded)))
    
# Remember to call main() so that the program executes!
if __name__ == '__main__':
    main()