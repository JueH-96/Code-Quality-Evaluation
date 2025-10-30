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
    X1 = int(data[idx])
    idx += 1
    
    trains = []
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        S = int(data[idx])
        idx += 1
        T = int(data[idx])
        idx += 1
        trains.append((A, B, S, T))
    
    # Create a graph with M+1 nodes (0 to M)
    graph = [[] for _ in range(M + 1)]
    
    # Add edge from super node (0) to node 1 with weight X1
    graph[0].append((1, X1))
    
    # Add edges from super node (0) to nodes 2 to M with weight 0
    for k in range(2, M + 1):
        graph[0].append((k, 0))
    
    # For each city, maintain a list of trains ending at that city, sorted by T_i
    from collections import defaultdict
    trains_by_end = defaultdict(list)
    for i in range(1, M + 1):
        _, B, _, T = trains[i - 1]
        trains_by_end[B].append((T, i))
    
    # Sort the lists by T_i
    for c in trains_by_end:
        trains_by_end[c].sort()
    
    # For each train j starting at c, find all i where B_i = A_j and T_i <= S_j
    for j in range(1, M + 1):
        A_j, _, S_j, _ = trains[j - 1]
        if A_j in trains_by_end:
            lst = trains_by_end[A_j]
            # Find all i where T_i <= S_j
            left = 0
            right = len(lst)
            # Binary search to find the rightmost i with T_i <= S_j
            while left < right:
                mid = (left + right) // 2
                if lst[mid][0] <= S_j:
                    left = mid + 1
                else:
                    right = mid
            # Add edges from i to j with weight T_i - S_j for all i in 0 to left-1
            for pos in range(left):
                T_i, i = lst[pos]
                weight = T_i - S_j
                graph[i].append((j, weight))
    
    # Dijkstra's algorithm to find shortest paths from node 0
    INF = float('inf')
    dist = [INF] * (M + 1)
    dist[0] = 0  # Super node
    
    heap = []
    heapq.heappush(heap, (0, 0))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    # Output X2 to XM
    result = [dist[k] for k in range(2, M + 1)]
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()