import collections
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency list: adj[u] contains a list of neighbors of u.
    # Users are 1-indexed, so arrays are of size N+1.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * (N + 1)  # To keep track of visited nodes during component traversal.
    total_new_edges = 0          # Accumulator for the final answer.
    
    # Iterate through all users to find connected components.
    for i in range(1, N + 1):
        if not visited[i]:
            # Node `i` is the first discovered node of a new connected component.
            # Perform BFS to find all nodes in this component and count its properties.
            
            current_component_nodes = 0
            # This will store the sum of degrees of nodes within this component.
            # Each edge (x,y) in the component contributes 1 to degree of x and 1 to degree of y.
            # So, (sum of degrees) / 2 gives the number of edges in the component.
            current_component_edges_sum_degrees = 0 
            
            q = collections.deque()
            
            # Initialize BFS with the starting node `i`.
            q.append(i)
            visited[i] = True # Mark node `i` as visited.
            
            # BFS traversal for the current component.
            while q:
                u = q.popleft() # Get a node from the queue to process.
                
                current_component_nodes += 1 # Increment node count for this component.
                # Add degree of u to sum. Degree is based on all its connections in the *original* graph.
                current_component_edges_sum_degrees += len(adj[u])
                
                # Add unvisited neighbors to the queue.
                for v_neighbor in adj[u]:
                    if not visited[v_neighbor]:
                        visited[v_neighbor] = True
                        q.append(v_neighbor)
            
            # After BFS, `current_component_nodes` is s_k (number of nodes in component k).
            # `current_component_edges_sum_degrees / 2` is e_k (number of edges in component k).
            s_k = current_component_nodes
            e_k = current_component_edges_sum_degrees // 2
            
            # A clique with s_k nodes has s_k * (s_k - 1) / 2 edges.
            edges_in_clique_k = s_k * (s_k - 1) // 2
            
            # Number of new edges (friendships) to add for this component to become a clique.
            new_edges_in_k = edges_in_clique_k - e_k
            total_new_edges += new_edges_in_k
            
    print(total_new_edges)

# Call the main solving function.
solve()