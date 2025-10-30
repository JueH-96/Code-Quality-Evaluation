def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    edges = [tuple(map(int, line.split())) for line in data[1:M+1]]
    
    if M == 0:
        print("Yes")
        print("1 " * N)
        return
    
    from collections import defaultdict
    
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check if the graph is bipartite
    color = {}
    
    def bfs(start):
        queue = [start]
        color[start] = 0  # Start coloring with 0
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
        return True
    
    for vertex in range(1, N + 1):
        if vertex not in color:
            if not bfs(vertex):
                print("No")
                return
    
    # If we reach here, the graph is bipartite
    # Assign values based on color
    result = [0] * N
    for vertex in range(1, N + 1):
        if color[vertex] == 0:
            result[vertex - 1] = 1  # Assign 1 to one color
        else:
            result[vertex - 1] = 2  # Assign 2 to the other color
    
    print("Yes")
    print(" ".join(map(str, result)))