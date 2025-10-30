import collections

def solve():
    n, m = map(int, input().split())
    friendships_input = []
    for _ in range(m):
        u, v = map(int, input().split())
        friendships_input.append(tuple(sorted((u, v))))
    
    adj = collections.defaultdict(list)
    initial_friendships_set = set()
    for u, v in friendships_input:
        adj[u].append(v)
        adj[v].append(u)
        initial_friendships_set.add(tuple(sorted((u, v))))
        
    visited = [False] * (n + 1)
    components = []
    for i in range(1, n + 1):
        if not visited[i]:
            component = []
            q = collections.deque([i])
            visited[i] = True
            component.append(i)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        component.append(v)
                        q.append(v)
            components.append(component)
            
    total_operations = 0
    for component in components:
        num_users_in_component = len(component)
        possible_edges_in_clique = num_users_in_component * (num_users_in_component - 1) // 2
        initial_edges_in_component = 0
        component_users_set = set(component)
        for u, v in friendships_input:
            if u in component_users_set and v in component_users_set:
                initial_edges_in_component += 1
                
        operations_in_component = possible_edges_in_clique - initial_edges_in_component
        total_operations += operations_in_component
        
    print(total_operations)

if __name__ == '__main__':
    solve()