# YOUR CODE HERE
def count_reachable_pairs(N, edges):
    graph = [[] for _ in range(N)]
    for i, v in enumerate(edges):
        graph[i].append(v - 1)

    visited = [False] * N
    stack = []
    component = []
    components = []

    def dfs(node):
        visited[node] = True
        stack.append(node)
        component.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
            elif neighbor in stack:
                start = stack.index(neighbor)
                components.append(stack[start:])
                return True
        
        stack.pop()
        return False

    for i in range(N):
        if not visited[i]:
            component = []
            if dfs(i):
                components.append(component)

    result = N  # Each vertex can reach itself
    for comp in components:
        size = len(comp)
        result += size * size

    return result

N = int(input())
edges = list(map(int, input().split()))

print(count_reachable_pairs(N, edges))