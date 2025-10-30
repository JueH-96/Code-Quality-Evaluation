import sys
import collections

def solve():
    N, X = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    # --- Step 1: Precompute P_inv and Q_inv for BFS ---
    # P_inv[k] stores the box 'i' such that P_i = k
    # Q_inv[k] stores the box 'i' such that Q_i = k
    P_inv = [0] * (N + 1)
    Q_inv = [0] * (N + 1)
    for i in range(N):
        P_inv[P[i]] = i + 1
        Q_inv[Q[i]] = i + 1

    # --- Step 2: Impossibility check based on ball flow cycles ---
    # For a ball originating in box 'k' to eventually reach box 'X',
    # 'X' must be part of the cycle containing 'k' in the respective graph (P for red, Q for blue).
    
    # Stores True if the cycle containing node 'i' in the P-graph includes X, False otherwise.
    node_cycle_contains_X_R = [False] * (N + 1)
    visited_R = [False] * (N + 1)
    
    # Iterate through all nodes to find all disjoint cycles in the P-graph
    for i in range(1, N + 1):
        if not visited_R[i]:
            path = [] # Stores nodes visited in the current traversal
            curr = i
            # Traverse until a visited node is encountered, indicating a cycle
            while not visited_R[curr]:
                visited_R[curr] = True
                path.append(curr)
                curr = P[curr - 1] # P is 0-indexed array, values are 1-indexed box numbers
            
            # 'curr' is the node where the cycle closes. Find its index in the current path.
            cycle_start_idx = path.index(curr)
            # The actual cycle nodes are from cycle_start_idx to the end of path
            cycle_nodes = set(path[cycle_start_idx:])
            
            # Check if X is in this cycle
            x_in_this_cycle = (X in cycle_nodes)
            
            # Mark all nodes that were part of this traversal (the cycle and any path leading to it)
            # with whether their cycle contains X.
            # For permutation graphs, the path *is* the cycle itself.
            for node in path:
                node_cycle_contains_X_R[node] = x_in_this_cycle
                
    # Repeat the same process for the Q-graph (blue balls)
    node_cycle_contains_X_B = [False] * (N + 1)
    visited_B = [False] * (N + 1)

    for i in range(1, N + 1):
        if not visited_B[i]:
            path = []
            curr = i
            while not visited_B[curr]:
                visited_B[curr] = True
                path.append(curr)
                curr = Q[curr - 1]
            
            cycle_start_idx = path.index(curr)
            cycle_nodes = set(path[cycle_start_idx:])
            
            x_in_this_cycle = (X in cycle_nodes)
            
            for node in path:
                node_cycle_contains_X_B[node] = x_in_this_cycle

    # Apply the impossibility check:
    # If any box 'i' (other than X) initially has balls, and those balls cannot reach X
    # through their defined flow path (because X is not in their cycle), then it's impossible.
    for i in range(1, N + 1):
        if i == X:
            continue # Box X is allowed to contain balls. Its own initial balls don't "need" to reach X.
        
        if A[i-1] > 0 and not node_cycle_contains_X_R[i]:
            print(-1)
            return
        
        if B[i-1] > 0 and not node_cycle_contains_X_B[i]:
            print(-1)
            return

    # --- Step 3: Calculate minimum operations using BFS on reverse dependencies ---
    # We count operations by performing a BFS starting from boxes that initially have balls
    # (and are not X). If a box 'curr' must be operated on, then the boxes that would send
    # balls to 'curr' (P_inv[curr] and Q_inv[curr]) must also be operated on if they are not X.
    
    ops_count = 0
    # to_operate_on[i] is True if box 'i' has been identified as needing an operation.
    to_operate_on = [False] * (N + 1) 
    q = collections.deque()

    # Initialize the queue with all boxes 'i' (that are not X) which initially contain balls.
    for i in range(1, N + 1):
        if i == X:
            continue
        if A[i-1] > 0 or B[i-1] > 0:
            if not to_operate_on[i]: # Avoid re-adding if a box qualifies for both A and B balls
                to_operate_on[i] = True
                q.append(i)

    # Perform BFS to propagate the 'need for operation'
    while q:
        curr = q.popleft()
        ops_count += 1

        # Check the boxes that would send balls to 'curr'
        # If P_inv[curr] (the box that sends red balls to 'curr') is not X
        # and has not yet been marked for operation, then it must be operated on.
        prev_P = P_inv[curr]
        if prev_P != X and not to_operate_on[prev_P]:
            to_operate_on[prev_P] = True
            q.append(prev_P)
        
        # Do the same for Q_inv[curr] (the box that sends blue balls to 'curr')
        prev_Q = Q_inv[curr]
        if prev_Q != X and not to_operate_on[prev_Q]:
            to_operate_on[prev_Q] = True
            q.append(prev_Q)
            
    print(ops_count)

solve()