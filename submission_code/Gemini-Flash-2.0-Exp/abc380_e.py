def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))

    def dfs(start_node, new_color):
        initial_color = colors[start_node - 1]
        colors[start_node - 1] = new_color
        
        stack = [start_node]
        visited = {start_node}
        
        while stack:
            node = stack.pop()
            
            # Check left neighbor
            if node > 1 and colors[node - 2] == initial_color and node - 1 not in visited:
                colors[node - 2] = new_color
                stack.append(node - 1)
                visited.add(node - 1)
            
            # Check right neighbor
            if node < n and colors[node] == initial_color and node + 1 not in visited:
                colors[node] = new_color
                stack.append(node + 1)
                visited.add(node + 1)

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            x, c = query[1], query[2]
            dfs(x, c)
        else:
            c = query[1]
            count = colors.count(c)
            print(count)

solve()