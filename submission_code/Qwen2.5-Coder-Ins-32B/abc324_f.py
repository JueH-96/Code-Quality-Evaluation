# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        b = int(data[index + 2])
        c = int(data[index + 3])
        edges.append((u, v, b, c))
        index += 4
    
    # Convert to adjacency list
    from collections import defaultdict
    adj_list = defaultdict(list)
    for u, v, b, c in edges:
        adj_list[u].append((v, b, c))
    
    # Use Dijkstra's algorithm with a max-heap to maximize beauty/cost ratio
    import heapq
    max_heap = [(-float('inf'), 0, 0)]  # (-beauty/cost, beauty, cost)
    visited = [False] * N
    max_ratio = -float('inf')
    
    while max_heap:
        neg_ratio, total_beauty, total_cost = heapq.heappop(max_heap)
        ratio = -neg_ratio
        
        if visited[N - 1]:
            break
        
        if visited[total_cost]:
            continue
        
        visited[total_cost] = True
        
        if total_cost == N - 1:
            max_ratio = max(max_ratio, ratio)
            continue
        
        for v, b, c in adj_list[total_cost]:
            new_beauty = total_beauty + b
            new_cost = total_cost + c
            new_ratio = new_beauty / new_cost
            heapq.heappush(max_heap, (-new_ratio, new_beauty, v))
    
    print(f"{max_ratio:.15f}")

if __name__ == "__main__":
    main()