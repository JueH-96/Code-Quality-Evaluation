import sys
from collections import deque

# Use faster input reading by default
input = sys.stdin.readline

# It's good practice to potentially increase recursion depth for competitive programming problems,
# especially those involving graphs or recursion, although BFS is iterative and doesn't rely on deep recursion stack.
# Setting it high ensures that even if some library functions internally use recursion, it won't fail easily.
# sys.setrecursionlimit(400000) 

def solve():
    # Read the number of vertices N
    N = int(input())
    
    # Adjacency list representation of the graph. Using 1-based indexing for vertices 1 to N.
    # Initialize a list of N+1 empty lists. adj[i] will store neighbors of vertex i.
    adj = [[] for _ in range(N + 1)]
    
    # Handle the base case where N=2.
    # In a tree with 2 vertices, say 1--v, both vertices have degree 1 and are leaves.
    if N == 2:
        # Read the edge information from input to consume the line, but its value is not needed.
        input() 
        # Since vertex 1 is a leaf, it can be deleted in the first operation.
        print(1)
        return

    # Read the N-1 edges of the tree and build the adjacency list.
    # Each edge (u, v) means u is adjacent to v, and v is adjacent to u.
    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # Check if vertex 1 is initially a leaf. 
    # A leaf vertex has degree at most 1. This covers degree 0 (isolated vertex) and degree 1.
    # For N >= 2, vertex 1 cannot be isolated initially in a connected tree. So check degree 1.
    if len(adj[1]) <= 1:
        # If vertex 1 is already a leaf, it can be chosen for deletion immediately.
        # The minimum number of operations is 1.
        print(1)
        return

    # The core idea is that to delete vertex 1, it must eventually become a leaf.
    # This happens when all but at most one of its neighbors are deleted (or rather, the entire subtrees connected through them).
    # To minimize operations, we should keep the largest subtree connected to vertex 1 intact,
    # delete all vertices in other subtrees connected to vertex 1, and then finally delete vertex 1.
    # Let T_i be the subtree rooted at neighbor n_i when the edge (1, n_i) is removed.
    # The minimum number of operations required is N - max(|T_i|) over all neighbors n_i of vertex 1.
    
    # Variable to store the maximum size found for any subtree T_i rooted at a neighbor of vertex 1.
    max_subtree_size = 0 
    
    # Get the list of direct neighbors of vertex 1.
    neighbors_of_1 = adj[1]
    
    # Iterate through each neighbor of vertex 1.
    for neighbor in neighbors_of_1:
        
        # Perform Breadth-First Search (BFS) starting from this 'neighbor'.
        # The goal is to find the size of the connected component containing 'neighbor'
        # *after* conceptually removing the edge (1, neighbor).
        # We achieve this by initializing the BFS exploration such that vertex 1 is treated as 'visited'
        # from the start, preventing the BFS from traversing back to vertex 1.
        
        q = deque([neighbor]) # Initialize the BFS queue with the starting neighbor.
        
        # 'visited' set keeps track of nodes visited during this specific BFS traversal.
        # Start by marking vertex 1 and the 'neighbor' as visited.
        visited = {1, neighbor} 
        
        # 'count' will store the number of vertices in the component found by this BFS.
        # Initialize to 1 because the 'neighbor' itself is counted.
        count = 1 
        
        # Standard BFS loop: continue as long as the queue is not empty.
        while q:
            curr = q.popleft() # Dequeue the next vertex to process.
            
            # Explore all adjacent vertices (neighbors) of the current vertex 'curr'.
            for child in adj[curr]:
                # If the adjacent vertex 'child' has not been visited yet in this BFS run:
                if child not in visited:
                    visited.add(child) # Mark 'child' as visited.
                    q.append(child)    # Add 'child' to the queue for processing later.
                    count += 1         # Increment the count of vertices in this component.
        
        # After the BFS from 'neighbor' completes, 'count' holds the size of the subtree T_neighbor.
        # Update 'max_subtree_size' if the size of the current subtree is the largest found so far.
        max_subtree_size = max(max_subtree_size, count)

    # The minimum number of operations is N (total vertices) minus the size of the largest subtree found.
    # This strategy ensures vertex 1 becomes a leaf (degree 1) connected to the largest subtree,
    # and then it can be deleted. The number of deletions needed is exactly the count of vertices
    # NOT in the largest subtree (excluding vertex 1 itself initially), plus one final operation for vertex 1.
    # This sum equals N - |T_max|.
    print(N - max_subtree_size)

# Execute the main function to solve the problem based on standard input.
solve()