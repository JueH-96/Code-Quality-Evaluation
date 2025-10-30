# YOUR CODE HERE
import sys
import heapq

def solve():
    N = int(sys.stdin.readline())
    # Read the sequence A. Adjust A to be 0-indexed internally, but represent node indices 1..N.
    # The input A = (A_1, ..., A_N) where 0 <= A_i < i.
    # A_list[i-1] is the value A_i given in the input for node i.
    A_list = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # adj[u] will store the list of nodes v such that there is an edge u -> v.
    # This means P_u < P_v is required.
    adj = [[] for _ in range(N + 1)] 
    # in_degree[v] stores the number of incoming edges to node v.
    in_degree = [0] * (N + 1)

    # Build the directed graph G based on the two conditions.
    # Nodes are indexed 1 to N.
    for i in range(1, N + 1): 
        # Get A_i for node i. Note A_list is 0-indexed.
        a_i = A_list[i-1] 
        
        # Condition 2: P_{A_i} < P_i if A_i > 0
        # This implies a directed edge from A_i to i: A_i -> i
        if a_i > 0:
            # Add edge from node a_i to node i
            adj[a_i].append(i)
            # Increment in-degree of the target node i
            in_degree[i] += 1
            
        # Condition 1: P_j > P_i for any integer j with A_i < j < i
        # This implies P_i < P_j for such j.
        # This implies directed edges from i to j: i -> j
        # Iterate through j values strictly between A_i and i
        for j in range(a_i + 1, i): 
             # Check if j is a valid node index (1 <= j <= N). 
             # Since j < i <= N, j is always <= N. Since A_i >= 0, a_i+1 >= 1, so j >= 1.
             # Add edge from node i to node j
             adj[i].append(j)
             # Increment in-degree of the target node j
             in_degree[j] += 1 

    # We need to count the number of topological sorts of the graph G.
    # The problem guarantees that at least one permutation exists, which implies G is a DAG.
    
    # A method sometimes cited for similar problems involves simulating Kahn's algorithm:
    # Maintain a set of source nodes (in-degree 0). At each step, multiply the total count
    # by the number of available sources. Then, pick a canonical node from the sources
    # (e.g., the one with the largest index) to remove and continue the simulation.
    
    # We use a max-heap to efficiently find the source node with the largest index.
    # Python's heapq is a min-heap, so we store negative indices to simulate a max-heap.
    pq = [] 
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            heapq.heappush(pq, -i) # Push negative index: -1 for node 1, -2 for node 2 etc.

    ans = 1
    processed_count = 0
    
    # The simulation runs for N steps, processing one node at each step.
    for _ in range(N):
       # Check if there are available source nodes.
       if not pq:
           # If pq is empty but not all nodes are processed, it implies a cycle.
           # However, the problem guarantees a valid permutation exists, so G is a DAG.
           # This case should ideally only be reached after N iterations when pq becomes empty.
           # If it occurs earlier, it indicates an error in logic or graph construction.
           # For robustness against unexpected issues:
           if processed_count != N:
               # This state indicates an issue if reached before N nodes are processed.
               # Output 0 as count, though this path shouldn't be taken given problem constraints.
               print(0) 
               return
           break # Normal termination after processing N nodes

       # Get the number of available source nodes (choices for the current step).
       current_available_count = len(pq)
       
       # Update the total count by multiplying with the number of choices.
       ans = (ans * current_available_count) % MOD
       
       # Select the node with the LARGEST index from available sources to proceed with the simulation.
       # This is achieved by popping the smallest element from the min-heap (which corresponds to the largest index).
       neg_u = heapq.heappop(pq)
       u = -neg_u # Convert back to the actual node index (positive).
       
       processed_count += 1

       # Process the selected node u: remove it conceptually from the graph
       # by updating the in-degrees of its neighbors.
       for v_neighbor in adj[u]:
             in_degree[v_neighbor] -= 1
             # If a neighbor's in-degree drops to 0, it becomes a new source node.
             if in_degree[v_neighbor] == 0:
                 # Add the new source node to the max-heap.
                 heapq.heappush(pq, -v_neighbor) 

    # Optional final check: ensure exactly N nodes were processed.
    # If not, it might indicate an issue, though problem guarantees should prevent this.
    if processed_count != N:
       print(0) # Indicates an error or unexpected state
       return

    # Print the final computed count modulo MOD.
    print(ans)

# Execute the solve function to run the program.
solve()