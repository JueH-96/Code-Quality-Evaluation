def find_strongest_programmer():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict, deque
    
    # Adjacency list for the graph
    graph = defaultdict(list)
    indegree = [0] * (N + 1)
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        graph[A].append(B)
        indegree[B] += 1
        index += 2
    
    # Topological sort using Kahn's algorithm
    zero_indegree = deque()
    for i in range(1, N + 1):
        if indegree[i] == 0:
            zero_indegree.append(i)
    
    topological_order = []
    
    while zero_indegree:
        node = zero_indegree.popleft()
        topological_order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree.append(neighbor)
    
    if len(topological_order) != N:
        print(-1)
        return
    
    # The last node in topological order is a candidate for the strongest programmer
    candidate = topological_order[-1]
    
    # Check if this candidate can reach all other nodes
    visited = [False] * (N + 1)
    
    def dfs(node):
        stack = [node]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
    
    dfs(candidate)
    
    if all(visited[1:]):
        print(candidate)
    else:
        print(-1)