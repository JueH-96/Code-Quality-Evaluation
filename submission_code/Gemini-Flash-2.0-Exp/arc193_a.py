def solve():
    N = int(input())
    W = list(map(int, input().split()))
    LR = []
    for _ in range(N):
        LR.append(list(map(int, input().split())))
    Q = int(input())
    queries = []
    for _ in range(Q):
        queries.append(list(map(int, input().split())))

    def are_intervals_disjoint(interval1, interval2):
        return interval1[1] < interval2[0] or interval2[1] < interval1[0]

    def build_graph(N, LR):
        graph = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if are_intervals_disjoint(LR[i], LR[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph

    graph = build_graph(N, LR)

    def find_min_weight_path(graph, start, end, weights):
        start -= 1
        end -= 1
        
        if start == end:
            return weights[start]

        queue = [(start, [start], weights[start])]
        visited = {start}
        min_weight = float('inf')
        
        while queue:
            node, path, current_weight = queue.pop(0)
            
            if node == end:
                min_weight = min(min_weight, current_weight)
                continue
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], current_weight + weights[neighbor]))
        
        if min_weight == float('inf'):
            return -1
        else:
            return min_weight

    for s, t in queries:
        result = find_min_weight_path(graph, s, t, W)
        print(result)

solve()