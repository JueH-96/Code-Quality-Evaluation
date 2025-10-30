import collections

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    guards = []
    for _ in range(k):
        p, h = map(int, input().split())
        guards.append({'position': p, 'stamina': h})
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    guarded_vertices = set()
    
    for guard in guards:
        start_vertex = guard['position']
        stamina = guard['stamina']
        distances = {}
        for i in range(1, n + 1):
            distances[i] = -1
        distances[start_vertex] = 0
        queue = collections.deque([start_vertex])
        
        while queue:
            current_vertex = queue.popleft()
            if distances[current_vertex] > stamina:
                continue
            guarded_vertices.add(current_vertex)
            for neighbor in adjacency_list[current_vertex]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current_vertex] + 1
                    queue.append(neighbor)
                    
    sorted_guarded_vertices = sorted(list(guarded_vertices))
    print(len(sorted_guarded_vertices))
    print(*(sorted_guarded_vertices))

if __name__ == '__main__':
    solve()