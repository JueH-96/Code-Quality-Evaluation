def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append({'x': x, 'y': y, 'index': _ + 1})
    
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        for j in range(i + 1, n):
            u_index = points[i]['index']
            v_index = points[j]['index']
            u_x = points[i]['x']
            u_y = points[i]['y']
            v_x = points[j]['x']
            v_y = points[j]['y']
            
            related = False
            if (u_x < v_x and u_y < v_y) or (u_x > v_x and u_y > v_y):
                related = True
            if (v_x < u_x and v_y < u_y) or (v_x > u_x and v_y > u_y):
                related = True
                
            if related:
                adj[u_index].append(v_index)
                adj[v_index].append(u_index)
                
    memo = {}
    
    def get_independent_set_count(vertex_set_tuple):
        vertex_set = tuple(sorted(vertex_set_tuple))
        if not vertex_set:
            return 1
        if vertex_set in memo:
            return memo[vertex_set]
        
        v = vertex_set[0]
        remaining_vertices_tuple = vertex_set[1:]
        remaining_vertices_set = set(remaining_vertices_tuple)
        
        # Case 1: Include vertex v in the independent set
        excluded_neighbors = set(adj[v])
        next_set_include_v = []
        for vertex in remaining_vertices_tuple:
            if vertex not in excluded_neighbors:
                next_set_include_v.append(vertex)
        count1 = get_independent_set_count(tuple(next_set_include_v))
        
        # Case 2: Exclude vertex v from the independent set
        count2 = get_independent_set_count(remaining_vertices_tuple)
        
        result = (count1 + count2) % 998244353
        memo[vertex_set] = result
        return result
        
    initial_balls = tuple(range(1, n + 1))
    count = get_independent_set_count(initial_balls)
    print(count)

if __name__ == '__main__':
    solve()