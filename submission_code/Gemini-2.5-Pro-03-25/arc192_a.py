# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep recursive calls if needed, though iterative DFS avoids this.
# Python's default recursion depth is typically 1000. For N=200000, 2N=400000 nodes,
# a deep recursion might exceed the default limit. Using iterative DFS is safer.
# try:
#     sys.setrecursionlimit(400010) # Set to 2*N + buffer
# except Exception: # Handle environments where setting recursion limit fails
#     pass 

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Problem Statement Analysis:
    # We are given a binary sequence A of length N. We want to determine if there exists
    # a string S of length N consisting of uppercase English letters such that we can 
    # make all elements of A equal to 1 using the specified operations.
    # The operations depend on substrings of S of length 3.
    # Op 1: If S_i S_{i+1} S_{i+2} = "ARC", replace A_i, A_{i+1} with 1.
    # Op 2: If S_i S_{i+1} S_{i+2} = "CRA", replace A_i, A_{i+1} with 1.
    # (Note: S_{i+2}=A, S_{i+1}=R, S_i=C is equivalent to S_i S_{i+1} S_{i+2} = "CRA").
    # Indices are modulo N.

    # Core Logic:
    # For an element A_k = 0 to eventually become 1, it must be affected by an operation.
    # A_k is affected by an operation at index i=k or i=k-1 (modulo N).
    # Operation at index i requires S_i S_{i+1} S_{i+2} to be "ARC" or "CRA".
    # Both "ARC" and "CRA" require the middle character S_{i+1} to be 'R'.
    # Therefore, an operation at index i is possible only if S_{i+1} = 'R'.
    #
    # Operation at index k-1 affects A_{k-1}, A_k. Requires S_k = 'R'.
    # Operation at index k affects A_k, A_{k+1}. Requires S_{k+1} = 'R'.
    #
    # So, a necessary condition for A_k=0 to become 1 is that either S_k = 'R' OR S_{k+1} = 'R'.
    # Let X_k be the boolean proposition "S_k = 'R'".
    # For each k where A_k = 0, we must satisfy the clause (X_k OR X_{k+1}). (Indices modulo N)
    #
    # This forms a 2-Satisfiability (2-SAT) problem. The question is whether there exists
    # an assignment of truth values to X_0, ..., X_{N-1} that satisfies all required clauses.
    # If this 2-SAT instance is unsatisfiable, then the necessary condition cannot be met,
    # implying no such string S exists.
    # We assume this necessary condition is also sufficient. That is, if the 2-SAT instance
    # is satisfiable, then there exists a good string S. This assumption is based on the idea
    # that if we can assign 'R's consistent with the necessary conditions, we have enough
    # freedom with other characters ('A', 'C', etc.) to enable the required operations without conflicts.

    # Implementation Strategy:
    # Model the 2-SAT problem using an implication graph.
    # Use Kosaraju's algorithm or Tarjan's algorithm to find Strongly Connected Components (SCCs).
    # Check if any variable X_k and its negation ~X_k lie in the same SCC.
    # If they do, the 2-SAT instance is unsatisfiable, output "No".
    # Otherwise, it is satisfiable, output "Yes".

    # Build the implication graph:
    # 2N nodes: 0 to N-1 represent literals X_0..X_{N-1}.
    # N to 2N-1 represent literals ~X_0..~X_{N-1}. Node k+N represents ~X_k.
    adj = [[] for _ in range(2 * N)]
    adj_rev = [[] for _ in range(2 * N)] # Adjacency list for the reversed graph

    # Helper function to get the node index for the negation of a literal.
    # Input k is a node index (0 to 2N-1).
    def neg(k):
        if k < N:
            return k + N  # Negation of X_k (node k) is ~X_k (node k+N)
        else:
            return k - N  # Negation of ~X_{k-N} (node k) is X_{k-N} (node k-N)

    num_clauses = 0
    for k in range(N):
        if A[k] == 0:
            num_clauses += 1
            
            # The clause is (X_k OR X_{k+1}), indices modulo N.
            k_node = k
            k_plus_1_node = (k + 1) % N
            
            # Convert clause to implications: (~X_k => X_{k+1}) AND (~X_{k+1} => X_k)
            neg_k_node = neg(k_node)             # Node representing ~X_k
            neg_k_plus_1_node = neg(k_plus_1_node) # Node representing ~X_{k+1}

            # Add edge for implication ~X_k => X_{k+1}
            adj[neg_k_node].append(k_plus_1_node)
            adj_rev[k_plus_1_node].append(neg_k_node) # Add reverse edge for Kosaraju's second pass
            
            # Add edge for implication ~X_{k+1} => X_k
            adj[neg_k_plus_1_node].append(k_node)
            adj_rev[k_node].append(neg_k_plus_1_node) # Add reverse edge

    # If A initially contains no 0s, any string S is good.
    if num_clauses == 0:
        print("Yes")
        return

    # Kosaraju's Algorithm for finding SCCs:
    
    order = [] # Stores nodes in order of finishing times (reverse post-order)
    visited = [False] * (2 * N)
    
    # First Pass: Perform DFS on the original graph `adj` to compute finishing order.
    # Use iterative DFS to avoid potential stack overflow on large N.
    for i in range(2 * N):
        if not visited[i]:
            # The stack stores tuples: (node, iterator over its neighbors)
            dfs_stack = [(i, iter(adj[i]))]
            # path_stack is auxiliary, helps track nodes currently being processed in DFS path
            path_stack = [i] 
            visited[i] = True

            while dfs_stack:
                curr_node, neighbors = dfs_stack[-1]
                 
                try:
                    # Try to get the next neighbor
                    next_node = next(neighbors)
                    if not visited[next_node]:
                        # If neighbor not visited, push it onto stack for processing
                        visited[next_node] = True
                        dfs_stack.append((next_node, iter(adj[next_node])))
                        path_stack.append(next_node) # Push onto path stack

                except StopIteration:
                    # No more neighbors for curr_node or all neighbors processed
                    # This means curr_node exploration is finished
                    finished_node = path_stack.pop() # Node finished, pop from path stack
                    order.append(finished_node) # Append to post-order list
                    dfs_stack.pop() # Pop from main DFS stack


    # Second Pass: Perform DFS on the reversed graph `adj_rev`, processing nodes
    # in the reverse order obtained from the first pass.
    scc = [-1] * (2 * N) # scc[i] will store the SCC ID for node i
    scc_count = 0
    
    visited = [False] * (2 * N) # Reset visited flags for the second pass

    for i in range(2 * N - 1, -1, -1):
        start_node = order[i] # Process nodes in reverse post-order
        if not visited[start_node]:
            # If node not yet assigned to an SCC, start DFS to find its SCC
            component_nodes = []
            # Use iterative DFS for the second pass as well
            stack = [start_node]
            visited[start_node] = True
            
            while stack:
                curr_node = stack.pop()
                component_nodes.append(curr_node)
                scc[curr_node] = scc_count # Assign SCC ID
                
                # Explore neighbors in the reversed graph
                for neighbor in adj_rev[curr_node]:
                    if not visited[neighbor]:
                         visited[neighbor] = True
                         stack.append(neighbor)
            
            scc_count += 1 # Increment SCC ID for the next component

    # Check for Satisfiability:
    # The 2-SAT instance is unsatisfiable iff there exists a variable X_k such that
    # X_k (node k) and ~X_k (node k+N) belong to the same SCC.
    for k in range(N):
        # Check if node k has been assigned an SCC ID (might not if graph is disconnected)
        # And check if its SCC ID is same as its negation's SCC ID.
        if scc[k] != -1 and scc[k] == scc[neg(k)]: 
             # Unsatisfiable condition found
             print("No")
             return

    # If the loop completes without finding any contradictions, the 2-SAT instance is satisfiable.
    print("Yes")

solve()