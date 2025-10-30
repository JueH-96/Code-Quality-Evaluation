from sys import stdin, stdout
import heapq

def main():
    input = stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    weights = list(map(int, data[index:index + N]))
    index += N
    
    intervals = []
    for i in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        intervals.append((L, R))
        index += 2
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for i in range(Q):
        s = int(data[index]) - 1
        t = int(data[index + 1]) - 1
        queries.append((s, t))
        index += 2
    
    # Build the graph based on the intervals
    graph = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            if not (intervals[i][1] < intervals[j][0] or intervals[j][1] < intervals[i][0]):
                # They intersect, so no edge
                continue
            # They do not intersect, so we can add an edge
            graph[i].append(j)
            graph[j].append(i)
    
    # Function to find the minimum weight path using Dijkstra's algorithm
    def dijkstra(start):
        min_weight = [float('inf')] * N
        min_weight[start] = weights[start]
        heap = [(weights[start], start)]  # (current_weight, vertex)
        
        while heap:
            current_weight, u = heapq.heappop(heap)
            
            if current_weight > min_weight[u]:
                continue
            
            for v in graph[u]:
                new_weight = current_weight + weights[v]
                if new_weight < min_weight[v]:
                    min_weight[v] = new_weight
                    heapq.heappush(heap, (new_weight, v))
        
        return min_weight
    
    # Process each query
    results = []
    
    for s, t in queries:
        min_weights_from_s = dijkstra(s)
        result = min_weights_from_s[t]
        results.append(result if result != float('inf') else -1)
    
    stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()