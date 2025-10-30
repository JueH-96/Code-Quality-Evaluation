import collections

def solve():
    n = int(input())
    if n == 1:
        c_values = list(map(int, input().split()))
        print(0)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    c_values = list(map(int, input().split()))
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    def get_distance(start_node, end_node):
        if start_node == end_node:
            return 0
        distance = [-1] * (n + 1)
        distance[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            u = queue.popleft()
            if u == end_node:
                return distance[end_node]
            for v in adjacency_list[u]:
                if distance[v] == -1:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        return -1

    def calculate_f(vertex_x):
        f_x = 0
        for i in range(1, n + 1):
            f_x += c_values[i-1] * get_distance(vertex_x, i)
        return f_x
        
    min_f_value = float('inf')
    result = -1
    
    for v in range(1, n + 1):
        f_v = calculate_f(v)
        if f_v < min_f_value:
            min_f_value = f_v
            result = f_v
            
    print(result)

if __name__ == '__main__':
    solve()