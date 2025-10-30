import sys
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    idx = 3

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(input_data[idx]); b = int(input_data[idx+1])
        idx += 2
        adj[a].append(b)
        adj[b].append(a)

    # Read guards info
    guards = []
    for _ in range(K):
        p = int(input_data[idx]); h = int(input_data[idx+1])
        idx += 2
        guards.append((p, h))

    # dist[v] = maximum "remaining stamina" found so far that can reach v
    dist = [-1] * (N+1)

    # Max-heap (store negative values to simulate max-heap using heapq)
    heap = []
    for p, h in guards:
        dist[p] = h
        # push negative because Python's heapq is a min-heap
        heapq.heappush(heap, (-h, p))

    # Multi-source "descending stamina" BFS using a priority queue
    while heap:
        neg_s, u = heapq.heappop(heap)
        s = -neg_s
        # If we no longer have the best stamina for this node, skip
        if s < dist[u]:
            continue
        # Explore neighbors with stamina-1
        if s > 0:
            for v in adj[u]:
                if s - 1 > dist[v]:
                    dist[v] = s - 1
                    heapq.heappush(heap, (-(s-1), v))

    # Collect all guarded vertices
    guarded_vertices = [i for i in range(1, N+1) if dist[i] >= 0]

    # Print results
    print(len(guarded_vertices))
    print(" ".join(map(str, guarded_vertices)))

# DO NOT FORGET TO CALL THE MAIN FUNCTION
if __name__ == "__main__":
    main()