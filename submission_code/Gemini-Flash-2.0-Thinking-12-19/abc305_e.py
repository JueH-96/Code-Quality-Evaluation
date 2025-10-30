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
        
    guarded_vertices_set = set()
    
    for guard in guards:
        start_vertex = guard['position']
        stamina = guard['stamina']
        
        queue = collections.deque([(start_vertex, 0)])
        visited_in_bfs = {start_vertex}
        
        while queue:
            current_vertex, distance = queue.popleft()
            if distance <= stamina:
                guarded_vertices_set.add(current_vertex)
                for neighbor in adjacency_list[current_vertex]:
                    if neighbor not in visited_in_bfs and distance + 1 <= stamina:
                        visited_in_bfs.add(neighbor)
                        queue.append((neighbor, distance + 1))
                        
    guarded_vertices_list = sorted(list(guarded_vertices_set))
    print(len(guarded_vertices_list))
    print(*(guarded_vertices_list))

if __name__ == '__main__':
    solve()