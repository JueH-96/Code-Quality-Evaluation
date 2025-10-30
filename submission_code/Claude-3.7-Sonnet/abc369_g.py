def main():
    N = int(input())
    edges = []
    for _ in range(N - 1):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    
    scores = compute_scores(N, edges)
    for score in scores:
        print(score)

def compute_scores(N, edges):
    # Create adjacency list representation of the tree
    graph = [[] for _ in range(N + 1)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Run a DFS to compute the parent and edge weight for each vertex
    parent = [0] * (N + 1)
    edge_weight = [0] * (N + 1)
    
    def dfs(u, p, w):
        parent[u] = p
        edge_weight[u] = w
        for v, weight in graph[u]:
            if v != p:
                dfs(v, u, weight)
    
    dfs(1, 0, 0)
    
    # Compute the scores for each K
    chosen = []
    edges_in_subtree = set()
    current_weight = 0
    scores = []
    
    for K in range(1, N + 1):
        if K <= N - 1:
            # Compute the contribution of each vertex
            best_contribution = -1
            best_vertex = -1
            
            for v in range(2, N + 1):
                if v not in chosen:
                    # Compute the contribution of v to the subtree
                    u = v
                    contribution = 0
                    
                    while u != 1:
                        if (u, parent[u]) not in edges_in_subtree and (parent[u], u) not in edges_in_subtree:
                            contribution += edge_weight[u]
                        u = parent[u]
                    
                    if contribution > best_contribution:
                        best_contribution = contribution
                        best_vertex = v
            
            # Update the subtree by adding edges from best_vertex to root
            u = best_vertex
            while u != 1:
                if (u, parent[u]) not in edges_in_subtree and (parent[u], u) not in edges_in_subtree:
                    edges_in_subtree.add((u, parent[u]))
                    edges_in_subtree.add((parent[u], u))  # Add both directions
                    current_weight += edge_weight[u]
                u = parent[u]
            
            chosen.append(best_vertex)
        
        scores.append(2 * current_weight)
    
    return scores

if __name__ == "__main__":
    main()