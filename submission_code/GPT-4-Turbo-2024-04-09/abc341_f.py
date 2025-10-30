def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    from collections import defaultdict
    
    graph = defaultdict(list)
    
    for _ in range(M):
        u = int(data[idx]) - 1
        idx += 1
        v = int(data[idx]) - 1
        idx += 1
        graph[u].append(v)
        graph[v].append(u)
    
    W = []
    for _ in range(N):
        W.append(int(data[idx]))
        idx += 1
    
    A = []
    for _ in range(N):
        A.append(int(data[idx]))
        idx += 1
    
    # We will use a greedy approach to simulate the process
    # We will use a priority queue to always process the vertex with the highest number of pieces
    import heapq
    
    # Priority queue of (-pieces, vertex) because heapq is a min-heap and we need max-heap behavior
    pq = []
    for i in range(N):
        if A[i] > 0:
            heapq.heappush(pq, (-A[i], i))
    
    operations = 0
    
    while pq:
        neg_pieces, x = heapq.heappop(pq)
        pieces = -neg_pieces
        
        # Process all pieces from this vertex
        while pieces > 0:
            # Perform the operation
            operations += 1
            pieces -= 1
            
            # Determine which neighbors to move pieces to
            for neighbor in graph[x]:
                if W[neighbor] < W[x]:
                    A[neighbor] += 1
                    heapq.heappush(pq, (-A[neighbor], neighbor))
    
    print(operations)

if __name__ == "__main__":
    main()