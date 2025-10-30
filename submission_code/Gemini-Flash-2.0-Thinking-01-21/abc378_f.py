import collections

def solve():
    n = int(input())
    if n < 3:
        print(0)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    degrees = collections.defaultdict(int)
    for i in range(1, n + 1):
        degrees[i] = len(adjacency_list[i])
        
    count = 0
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if v in adjacency_list[u]:
                continue
            if degrees[u] != 2 or degrees[v] != 2:
                continue
                
            parent_map = {}
            queue = collections.deque([u])
            visited = {u}
            parent_map[u] = None
            path_found = False
            while queue:
                current_node = queue.popleft()
                if current_node == v:
                    path_found = True
                    break
                for neighbor in adjacency_list[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent_map[neighbor] = current_node
                        queue.append(neighbor)
                        
            if path_found:
                path = []
                current = v
                while current is not None:
                    path.append(current)
                    current = parent_map[current]
                path = path[::-1]
                
                valid_path_degrees = True
                if len(path) <= 2:
                    valid_path_degrees = True
                else:
                    for i in range(1, len(path) - 1):
                        if degrees[path[i]] != 3:
                            valid_path_degrees = False
                            break
                            
                if valid_path_degrees:
                    count += 1
                    
    print(count)

if __name__ == '__main__':
    solve()