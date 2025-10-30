# YOUR CODE HERE
import sys
import collections

# Function to solve the problem
def solve():
    # Read N (number of people) and M (number of insurance policies) from input
    N, M = map(int, sys.stdin.readline().split())

    # Initialize adjacency list for the tree structure
    # adj[i] contains list of children of node i (person i)
    # We use a list of lists for efficiency. Index 0 is unused, indices 1 to N correspond to people.
    adj = [[] for _ in range(N + 1)]
    
    # Read parent information if N > 1 (if N=1, there are no parents to read)
    if N > 1:
        # Read parents p_2, ..., p_N. The list contains N-1 integers.
        parents = list(map(int, sys.stdin.readline().split()))
        # Build adjacency list based on parent information
        # The i-th element in `parents` (0-indexed) is the parent of person i+2.
        for i in range(N - 1):
            child = i + 2  # Person ID of the child
            parent = parents[i] # Person ID of the parent
            # Append the child to the parent's list of children
            adj[parent].append(child)

    # Initialize depth array to store the depth of each person in the family tree.
    # depth[i] stores the depth of person i. Root (person 1) is at depth 0.
    # Initialize with -1, indicating that the depth has not been computed yet or the person is unreachable.
    depth = [-1] * (N + 1)
    
    # Queue for Breadth-First Search (BFS) to compute depths
    q_depth = collections.deque()
    # Track visited nodes during the depth calculation BFS using a boolean array
    processed_depth = [False] * (N + 1)
    
    # Start BFS from the root node (person 1) if N >= 1
    if N >= 1:
        q_depth.append((1, 0)) # Add root (person 1, depth 0) to the queue
        depth[1] = 0           # Set the depth of the root
        processed_depth[1] = True # Mark the root as visited

        # Perform BFS to calculate depths
        while q_depth:
            u, d = q_depth.popleft() # Get the current person u and their depth d
            
            # Iterate over the children v of person u
            for v in adj[u]:
               # If child v has not been visited yet
               if not processed_depth[v]:
                    depth[v] = d + 1          # Set the depth of child v
                    processed_depth[v] = True # Mark child v as visited
                    q_depth.append((v, d + 1)) # Add child v to the queue for further exploration

    # Initialize an array to store the maximum coverage depth initiated at each person.
    # `initial_max_depth[x]` will store the maximum value of `depth[x] + y` over all policies (x, y) purchased by person x.
    # Initialize with -1, representing no coverage initiated from this node yet.
    initial_max_depth = [-1] * (N + 1)

    # Process M insurance policies
    for _ in range(M):
        # Read the person x who bought the policy and the coverage duration y (generations)
        x, y = map(int, sys.stdin.readline().split())
        
        # Check if person x is valid (within 1 to N) and reachable from the root (depth is computed)
        if 1 <= x <= N and depth[x] != -1:
             # Update the maximum depth covered by policies starting at person x.
             # The policy covers person x and descendants up to y generations down.
             # The maximum depth reached by this coverage is depth[x] + y.
             initial_max_depth[x] = max(initial_max_depth[x], depth[x] + y)

    # Initialize a boolean array to track if each person is covered by at least one policy
    is_covered = [False] * (N + 1) 
    
    # Queue for BFS propagation of coverage information down the tree
    q_propagate = collections.deque()
    # Track visited nodes during the propagation BFS using a boolean array
    processed_propagate = [False] * (N + 1)
    
    # Start the propagation BFS from the root node (person 1) if N >= 1
    if N >= 1:
      # The root node 1 has no ancestors, so its inherited maximum coverage depth is -1.
      q_propagate.append((1, -1)) # Add (node, inherited_max_depth) tuple to the queue
      processed_propagate[1] = True # Mark the root as visited for propagation

      # Perform BFS propagation
      while q_propagate:
          u, inherited_max_depth = q_propagate.popleft() # Get current person u and the inherited depth limit from ancestor policies
          
          # Compute the effective maximum coverage depth for person u.
          # This is the maximum of the depth limit imposed by policies bought by u itself (`initial_max_depth[u]`)
          # and the depth limit inherited from its ancestors (`inherited_max_depth`).
          current_effective_max_depth = max(initial_max_depth[u], inherited_max_depth)
          
          # Check if person u is covered by any policy.
          # Person u is covered if they are reachable (depth[u] != -1) and their depth
          # does not exceed the effective maximum coverage depth applicable to them.
          if depth[u] != -1 and depth[u] <= current_effective_max_depth:
              is_covered[u] = True # Mark person u as covered

          # Propagate the coverage information to the children of person u
          for v in adj[u]:
               # If child v has not been processed in this propagation pass yet
               if not processed_propagate[v]:
                    processed_propagate[v] = True # Mark child v as processed
                    # Add child v to the queue. The child inherits the effective max depth limit computed for its parent u.
                    q_propagate.append((v, current_effective_max_depth))

    # Count the total number of covered people (persons 1 to N)
    # Sum the boolean values in the `is_covered` array for indices 1 to N (True counts as 1, False as 0)
    final_count = sum(is_covered[1:]) 
    # Print the final count to standard output
    print(final_count)

# Call the solve function to execute the logic
solve()