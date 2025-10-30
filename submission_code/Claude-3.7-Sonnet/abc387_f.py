# YOUR CODE HERE
import sys

MOD = 998244353

def find_sccs(graph, n):
    # Kosaraju's algorithm for finding strongly connected components
    visited = [False] * n
    order = []
    
    # First DFS to get finishing order
    def dfs1(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs1(neighbor)
        order.append(node)
    
    for i in range(n):
        if not visited[i]:
            dfs1(i)
    
    # Create reversed graph
    reversed_graph = [[] for _ in range(n)]
    for i in range(n):
        for j in graph[i]:
            reversed_graph[j].append(i)
    
    # Second DFS to find SCCs
    visited = [False] * n
    components = []
    
    def dfs2(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in reversed_graph[node]:
            if not visited[neighbor]:
                dfs2(neighbor, component)
    
    # Process nodes in reverse finishing order
    order.reverse()
    for node in order:
        if not visited[node]:
            component = []
            dfs2(node, component)
            components.append(component)
    
    return components

def solve(N, M, A):
    # Construct the graph
    graph = [[] for _ in range(N)]
    for i in range(N):
        graph[i].append(A[i] - 1)  # Adjust for 0-indexing
    
    # Find SCCs
    sccs = find_sccs(graph, N)
    
    # Map from node to its SCC index
    node_to_scc = [0] * N
    for idx, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = idx
    
    # Construct the condensed graph (DAG of SCCs)
    condensed_graph = [[] for _ in range(len(sccs))]
    reverse_condensed_graph = [[] for _ in range(len(sccs))]
    
    for node in range(N):
        scc_i = node_to_scc[node]
        neighbor = A[node] - 1
        scc_j = node_to_scc[neighbor]
        if scc_i != scc_j and scc_j not in condensed_graph[scc_i]:
            condensed_graph[scc_i].append(scc_j)
            reverse_condensed_graph[scc_j].append(scc_i)
    
    # DP calculation
    dp_cache = {}
    
    def compute_dp(scc, val):
        if (scc, val) in dp_cache:
            return dp_cache[(scc, val)]
        
        # For a sink SCC, there's 1 way to assign value val
        if not condensed_graph[scc]:
            return 1
        
        result = 1
        for next_scc in condensed_graph[scc]:
            # Sum of ways to assign values from 1 to val to next_scc
            ways = 0
            for next_val in range(1, val + 1):
                ways = (ways + compute_dp(next_scc, next_val)) % MOD
            result = (result * ways) % MOD
        
        dp_cache[(scc, val)] = result
        return result
    
    # Find root SCCs (those with no incoming edges)
    roots = [i for i in range(len(sccs)) if not reverse_condensed_graph[i]]
    
    # Calculate total number of valid sequences
    answer = 0
    for root in roots:
        for val in range(1, M + 1):
            answer = (answer + compute_dp(root, val)) % MOD
    
    return answer

def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    print(solve(N, M, A))

if __name__ == "__main__":
    main()