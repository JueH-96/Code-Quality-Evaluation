import sys

def dfs(graph, node, color, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if graph[node][neighbor] == color and neighbor not in visited:
            dfs(graph, neighbor, color, visited)

def solve():
    N, Q = map(int, input().split())
    graph = {i: {i+1: i} for i in range(1, N)}
    
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            x, c = map(int, query[1:])
            visited = set()
            dfs(graph, x, graph[x][x+1], visited)
            for node in visited:
                graph[node] = {i: c for i in graph[node]}
        else:
            c = int(query[1])
            count = 0
            for i in range(1, N+1):
                if graph[i][i] == c:
                    count += 1
            print(count)

if __name__ == "__main__":
    solve()