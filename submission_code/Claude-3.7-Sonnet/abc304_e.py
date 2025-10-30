from collections import deque

def compute_component_ids(graph, N):
    """Compute the component ID for each node."""
    component_ids = [0] * (N + 1)
    component_count = 0
    
    for node in range(1, N + 1):
        if component_ids[node] == 0:
            component_count += 1
            
            # BFS
            queue = deque([node])
            component_ids[node] = component_count
            
            while queue:
                curr_node = queue.popleft()
                
                for neighbor in graph[curr_node]:
                    if component_ids[neighbor] == 0:
                        component_ids[neighbor] = component_count
                        queue.append(neighbor)
    
    return component_ids

def main():
    # Read input
    N, M = map(int, input().split())
    
    # Initialize graph
    graph = [[] for _ in range(N + 1)]
    
    # Read edges
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)  # Undirected graph
    
    # Compute component IDs
    component_ids = compute_component_ids(graph, N)
    
    # Read forbidden path conditions
    K = int(input())
    forbidden_pairs = []
    for _ in range(K):
        x, y = map(int, input().split())
        forbidden_pairs.append((x, y))
    
    # Read questions
    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        
        # Check if adding the edge (p, q) would violate any forbidden pair
        is_good = True
        for x, y in forbidden_pairs:
            if ((component_ids[p] == component_ids[x] and component_ids[q] == component_ids[y]) or 
                (component_ids[p] == component_ids[y] and component_ids[q] == component_ids[x])):
                is_good = False
                break
        
        # Print the result
        print("Yes" if is_good else "No")

if __name__ == "__main__":
    main()