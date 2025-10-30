def main():
    import sys
    sys.setrecursionlimit(10000)
    N = int(sys.stdin.readline())
    edges = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Build adjacency list for the new graph
    from collections import defaultdict
    graph = defaultdict(list)
    edge_list = edges  # each edge is a string like 'AB'
    for i in range(len(edge_list)):
        for j in range(len(edge_list)):
            if i != j:
                e1 = edge_list[i]
                e2 = edge_list[j]
                if e1[1] == e2[0]:
                    graph[e1].append(e2)
    
    # Now build bipartite graph and find maximum matching
    # Left side is the same as edge_list, right side is also the same
    # We need to map each edge to an index for easier handling
    edge_to_idx = {e:i for i, e in enumerate(edge_list)}
    # Build adjacency list for bipartite graph
    adj = [[] for _ in range(len(edge_list))]
    for e1 in edge_list:
        for e2 in graph[e1]:
            adj[edge_to_idx[e1]].append(edge_to_idx[e2])
    
    # Maximum bipartite matching using DFS
    match_to = [-1] * len(edge_list)  # right to left
    def dfs(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_to[v] == -1 or dfs(match_to[v], visited):
                    match_to[v] = u
                    return True
        return False
    
    result = 0
    for u in range(len(edge_list)):
        visited = [False] * len(edge_list)
        if dfs(u, visited):
            result += 1
    
    print(len(edge_list) - result)

if __name__ == "__main__":
    main()