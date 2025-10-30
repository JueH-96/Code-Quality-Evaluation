def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = []
    for i in range(1, 2*N - 1, 2):
        u = int(data[i]) - 1
        v = int(data[i+1]) - 1
        edges.append((u, v))
    
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # DFS to compute the number of direct children
    def dfs(current, parent):
        size[current] = 1
        for neighbor in adj[current]:
            if neighbor == parent:
                continue
            size[current] += dfs(neighbor, current)
        return size[current]
    
    size = [0] * N
    dfs(0, -1)
    
    # Compute the number of direct children for each vertex
    direct_children = [0] * N
    for u in range(N):
        for v in adj[u]:
            if size[v] < size[u]:
                direct_children[u] += 1
    
    # For each vertex, sort its direct children's degrees in descending order
    children_degrees = []
    for u in range(N):
        degrees = []
        for v in adj[u]:
            if size[v] < size[u]:
                degrees.append(direct_children[v])
        children_degrees.append(sorted(degrees, reverse=True))
    
    max_snowflake_size = 0
    for u in range(N):
        degrees = children_degrees[u]
        num_children = len(degrees)
        for x in range(1, num_children + 1):
            y = degrees[x - 1]
            if y == 0:
                break
            snowflake_size = 1 + x + x * y
            if snowflake_size > max_snowflake_size:
                max_snowflake_size = snowflake_size
    
    print(N - max_snowflake_size)

if __name__ == "__main__":
    main()