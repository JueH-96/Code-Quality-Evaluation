import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    bridges = []
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        t = int(data[idx+2])
        bridges.append((u, v, t))
        idx += 3
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        K = int(data[idx])
        idx += 1
        B = list(map(int, data[idx:idx+K]))
        queries.append(B)
        idx += K
    # Precompute shortest paths from 1 and to N
    # Dijkstra from 1
    dist_from_1 = [float('inf')] * (N+1)
    dist_from_1[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist_from_1[u]:
            continue
        for bridge in bridges:
            if bridge[0] == u:
                v = bridge[1]
                t = bridge[2]
                if dist_from_1[v] > current_dist + t:
                    dist_from_1[v] = current_dist + t
                    heapq.heappush(heap, (dist_from_1[v], v))
            if bridge[1] == u:
                v = bridge[0]
                t = bridge[2]
                if dist_from_1[v] > current_dist + t:
                    dist_from_1[v] = current_dist + t
                    heapq.heappush(heap, (dist_from_1[v], v))
    # Dijkstra to N
    dist_to_N = [float('inf')] * (N+1)
    dist_to_N[N] = 0
    heap = []
    heapq.heappush(heap, (0, N))
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist_to_N[u]:
            continue
        for bridge in bridges:
            if bridge[0] == u:
                v = bridge[1]
                t = bridge[2]
                if dist_to_N[v] > current_dist + t:
                    dist_to_N[v] = current_dist + t
                    heapq.heappush(heap, (dist_to_N[v], v))
            if bridge[1] == u:
                v = bridge[0]
                t = bridge[2]
                if dist_to_N[v] > current_dist + t:
                    dist_to_N[v] = current_dist + t
                    heapq.heappush(heap, (dist_to_N[v], v))
    # Process each query
    for query in queries:
        K = len(query)
        # For each bridge in the query, find the minimum path that includes it
        # The path can be: 1 -> u -> v -> N or 1 -> v -> u -> N
        # So the total time is dist_from_1[u] + t + dist_to_N[v] or dist_from_1[v] + t + dist_to_N[u]
        # We need to choose the minimum among all these possibilities
        min_time = float('inf')
        for b in query:
            u, v, t = bridges[b-1]
            option1 = dist_from_1[u] + t + dist_to_N[v]
            option2 = dist_from_1[v] + t + dist_to_N[u]
            min_time = min(min_time, option1, option2)
        # Now, we need to ensure that all bridges are used
        # Since K can be up to 5, we can try all possible orders
        # But for K=1, it's already handled
        if K == 1:
            print(min_time)
            continue
        # For K > 1, we need to find a path that uses all bridges
        # We can model this as a path that visits all the bridges in some order
        # Since K is small, we can try all permutations
        from itertools import permutations
        # Get all the bridges in the query
        bridge_list = [bridges[b-1] for b in query]
        # Try all permutations of the bridges
        for perm in permutations(bridge_list):
            total_time = 0
            current_node = 1
            for bridge in perm:
                u, v, t = bridge
                if current_node == u:
                    total_time += t
                    current_node = v
                elif current_node == v:
                    total_time += t
                    current_node = u
                else:
                    # Need to find a path from current_node to u or v
                    # Choose the one with the minimum distance
                    option1 = dist_from_1[current_node] + dist_from_1[u] + t + dist_to_N[v]
                    option2 = dist_from_1[current_node] + dist_from_1[v] + t + dist_to_N[u]
                    if option1 < option2:
                        total_time += dist_from_1[current_node] + dist_from_1[u] + t
                        current_node = v
                    else:
                        total_time += dist_from_1[current_node] + dist_from_1[v] + t
                        current_node = u
            # After using all bridges, need to reach N
            total_time += dist_to_N[current_node]
            min_time = min(min_time, total_time)
        print(min_time)

if __name__ == "__main__":
    main()