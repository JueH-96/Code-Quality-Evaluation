import collections

def solve():
    n = int(input())
    a_list = list(map(int, input().split()))
    w_list = list(map(int, input().split()))
    
    cost_matrix = []
    for i in range(n):
        row = []
        for j in range(1, n + 1):
            if j == a_list[i]:
                row.append(0)
            else:
                row.append(w_list[i])
        cost_matrix.append(row)
        
    source = 0
    sink = 2 * n + 1
    capacity = collections.defaultdict(int)
    cost = collections.defaultdict(int)
    graph = collections.defaultdict(list)
    
    for i in range(n):
        item_node = i + 1
        box_node = n + i + 1
        graph[source].append(item_node)
        graph[item_node].append(source)
        capacity[source, item_node] = 1
        cost[source, item_node] = 0
        cost[item_node, source] = 0
        
        graph[box_node].append(sink)
        graph[sink].append(box_node)
        capacity[box_node, sink] = 1
        cost[box_node, sink] = 0
        cost[sink, box_node] = 0
        
        for j in range(n):
            box_index = j + 1
            graph[item_node].append(n + box_index)
            graph[n + box_index].append(item_node)
            capacity[item_node, n + box_index] = 1
            cost[item_node, n + box_index] = cost_matrix[i][j]
            cost[n + box_index, item_node] = -cost_matrix[i][j]
            
    min_total_cost = 0
    flow = 0
    
    while flow < n:
        distance = {i: float('inf') for i in range(sink + 1)}
        parent_edge = {i: None for i in range(sink + 1)}
        distance[source] = 0
        in_queue = {source}
        queue = collections.deque([source])
        
        while queue:
            u = queue.popleft()
            in_queue.remove(u)
            for v in graph[u]:
                if capacity[u, v] > 0 and distance[v] > distance[u] + cost[u, v]:
                    distance[v] = distance[u] + cost[u, v]
                    parent_edge[v] = (u, v)
                    if v not in in_queue:
                        queue.append(v)
                        in_queue.add(v)
                        
        if distance[sink] == float('inf'):
            break
            
        path_flow = float('inf')
        v = sink
        while v != source:
            u, _ = parent_edge[v]
            path_flow = min(path_flow, capacity[u, v])
            v = u
            
        flow += path_flow
        v = sink
        while v != source:
            u, _ = parent_edge[v]
            capacity[u, v] -= path_flow
            capacity[v, u] += path_flow
            min_total_cost += path_flow * cost[u, v]
            v = u
            
    print(min_total_cost)

if __name__ == '__main__':
    solve()