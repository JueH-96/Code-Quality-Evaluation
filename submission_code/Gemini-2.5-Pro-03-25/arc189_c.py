# YOUR CODE HERE
import sys
from collections import deque

# Set higher recursion depth limit for safety, although iterative approach is used mostly.
# sys.setrecursionlimit(2 * 10**5 + 10) # Not strictly necessary for this iterative solution.

def solve():
    # Read input N and X (target box index)
    N, X_1_based = map(int, sys.stdin.readline().split())
    # Convert X to 0-based index for internal use
    X = X_1_based - 1 

    # Read initial ball counts A (red) and B (blue) for each box
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Read permutations P and Q (destination boxes for red and blue balls respectively)
    P_1_based = list(map(int, sys.stdin.readline().split()))
    Q_1_based = list(map(int, sys.stdin.readline().split()))

    # Convert P and Q permutations to use 0-based indices
    P = [p - 1 for p in P_1_based]
    Q = [q - 1 for q in Q_1_based]

    # Compute S_min: the minimum set of boxes that must be operated on at least once.
    # Start with boxes that initially contain balls and are not the target box X.
    # Any such box must be operated on to move its balls out.
    S_init = {i for i in range(N) if (A[i] > 0 or B[i] > 0) and i != X}
    
    # S_min will store the final set of boxes that must be operated on.
    S_min = set(S_init) 
    # Use a queue for BFS to find all boxes reachable from S_init through P and Q paths, avoiding X.
    queue = deque(S_init) 
    
    # Keep track of nodes that have been added to the queue to avoid redundant processing in BFS.
    processed_bfs = set(S_init) 

    # Perform BFS to expand S_min.
    # If a box `curr` must be operated on (is in S_min), and it sends balls to `p_next` (via P)
    # or `q_next` (via Q), and these next boxes are not X, then `p_next` and `q_next`
    # must also potentially be operated on to forward the balls eventually towards X.
    while queue:
        curr = queue.popleft()
        
        # Check propagation via P mapping (for red balls)
        p_next = P[curr]
        # If the destination box for red balls is not X and is not already in S_min
        if p_next != X and p_next not in S_min:
            S_min.add(p_next) # Add it to the set of required operations
            # If this node hasn't been queued before, add it for processing.
            if p_next not in processed_bfs: 
                 queue.append(p_next)
                 processed_bfs.add(p_next) # Mark as queued
        
        # Check propagation via Q mapping (for blue balls)
        q_next = Q[curr]
        # If the destination box for blue balls is not X and is not already in S_min
        if q_next != X and q_next not in S_min:
            S_min.add(q_next) # Add it to the set of required operations
            # If this node hasn't been queued before, add it for processing.
            if q_next not in processed_bfs: 
                queue.append(q_next)
                processed_bfs.add(q_next) # Mark as queued


    # After computing S_min, check if it's possible to move all balls to X.
    possible = True # Flag to track if the goal state is achievable
    
    # Check paths for all initially present Red balls (using P transitions)
    # memo_P stores the computed destination for balls starting at node i:
    # - destination node index if the path terminates correctly at a node outside S_min
    # - -1 if the path enters a cycle consisting entirely of nodes within S_min (impossible state)
    memo_P = {} 

    for i in range(N):
        # Only need to check paths for boxes that initially contain red balls
        if A[i] > 0:
            # Check memoization table first: if destination for node i is already known
            if i in memo_P:
                dest = memo_P[i]
                # If the known destination is invalid (cycle detected or not X)
                if dest == -1 or dest != X:
                    possible = False # Goal is impossible
                    break # Stop checking further
                continue # Destination is X, valid. Continue to the next box check.

            # If destination not memoized, trace the path iteratively
            curr = i
            path_list = [] # Keep track of nodes visited in the current path trace
            path_trace_visited_set = set() # Used to detect cycles within this specific trace
            
            computed_dest = -2 # Sentinel value indicating destination not yet computed

            while True:
                # Check memoization table for the current node in the path
                if curr in memo_P: 
                    computed_dest = memo_P[curr] # Found precomputed result
                    break
                
                # Detect cycle by checking if current node was already visited in this trace
                if curr in path_trace_visited_set: 
                    computed_dest = -1 # Cycle detected, means path never terminates
                    
                    # Mark all nodes within the detected cycle as having destination -1 in the memo table
                    cycle_start_idx = -1
                    try:
                        # Find the first occurrence of 'curr' in path_list to identify cycle start
                        cycle_start_idx = path_list.index(curr)
                        # Mark all nodes from cycle start onwards
                        for k in range(cycle_start_idx, len(path_list)):
                             memo_P[path_list[k]] = -1
                    except ValueError:
                         # This case should not happen if cycle detection logic is correct
                         pass 

                    break # Exit loop after detecting cycle and marking memo

                # Check if path terminates: reached a node that is NOT in S_min
                # Such a node will not be operated on, so balls stop here.
                if curr not in S_min: 
                    computed_dest = curr # Destination found
                    break

                # If path continues: add current node to path history and move to the next node via P
                path_trace_visited_set.add(curr)
                path_list.append(curr)
                curr = P[curr]

            # After path trace finishes (found destination or cycle), update memoization table for all nodes visited in this path
            # All nodes in path_list lead to the same final destination status (either a node index or -1 for cycle)
            for node in path_list:
                 # Update memo only if the node's destination isn't already determined (e.g., if it was part of the cycle marked earlier)
                 if node not in memo_P:
                     memo_P[node] = computed_dest

            # Final check for the ball that started at box i:
            # The computed destination must be the target box X. If it's a cycle (-1) or a different box, it's impossible.
            if computed_dest == -1 or computed_dest != X:
                 possible = False # Goal is impossible
                 break # Stop checking further
    
    # If any red ball path check determined impossibility, print -1 and exit
    if not possible:
        print("-1")
        # sys.exit() # Use return instead of exit in function context
        return

    # Check paths for all initially present Blue balls (using Q transitions)
    # Logic is symmetric to the check for red balls.
    memo_Q = {} # Memoization table for blue ball paths

    for i in range(N):
        # Only need to check paths for boxes initially containing blue balls
         if B[i] > 0:
            # Check memoization table first
            if i in memo_Q:
                dest = memo_Q[i]
                if dest == -1 or dest != X:
                    possible = False
                    break
                continue

            # Trace path iteratively if destination not memoized
            curr = i
            path_list = []
            path_trace_visited_set = set()
            computed_dest = -2 # Sentinel value

            while True:
                # Check memoization table for current node
                if curr in memo_Q:
                    computed_dest = memo_Q[curr]
                    break
                
                # Detect cycle
                if curr in path_trace_visited_set:
                    computed_dest = -1 # Cycle detected
                    
                    # Mark cycle nodes in memo
                    cycle_start_idx = -1
                    try:
                        cycle_start_idx = path_list.index(curr)
                        for k in range(cycle_start_idx, len(path_list)):
                             memo_Q[path_list[k]] = -1
                    except ValueError:
                        pass

                    break # Exit loop after cycle detection

                # Check if path terminates (node not in S_min)
                if curr not in S_min:
                    computed_dest = curr # Destination found
                    break

                # Continue tracing via Q
                path_trace_visited_set.add(curr)
                path_list.append(curr)
                curr = Q[curr]

            # Update memoization table for nodes visited in this trace
            for node in path_list:
                if node not in memo_Q:
                   memo_Q[node] = computed_dest

            # Check if the computed destination is valid (must be X)
            if computed_dest == -1 or computed_dest != X:
                 possible = False
                 break # Stop checking further

    # Final output based on whether all checks passed
    if not possible:
         print("-1") # Goal is impossible
    else:
         # If possible, the minimum number of operations needed is the size of the set S_min.
         # This assumes "number of operations" means the count of distinct boxes operated on at least once.
         print(len(S_min))

# Call the main solver function
solve()