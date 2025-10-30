import collections

def solve():
    N, X_orig = map(int, input().split())
    A_balls = list(map(int, input().split())) # Using A_balls to avoid conflict with P, Q variable names if they were A, B
    B_balls = list(map(int, input().split()))
    P_orig = list(map(int, input().split()))
    Q_orig = list(map(int, input().split()))

    X = X_orig - 1 
    P = [val - 1 for val in P_orig]
    Q = [val - 1 for val in Q_orig]

    # Step 1: Determine "good" boxes
    is_good = [False] * N
    # num_pending_good_conditions[k]: how many successors of k (P[k], Q[k]) are not yet confirmed good.
    # k becomes good if this count reaches 0.
    num_pending_good_conditions = [2] * N 
    
    # Predecessor lists: rev_P_edges[j] contains k such that P[k]=j
    rev_P_edges = [[] for _ in range(N)]
    rev_Q_edges = [[] for _ in range(N)]
    for k_idx in range(N):
        rev_P_edges[P[k_idx]].append(k_idx)
        rev_Q_edges[Q[k_idx]].append(k_idx)

    q_good_bfs = collections.deque()
    
    # Initialize for X: X is good by definition.
    if not is_good[X]: # Should always be true here
        is_good[X] = True
        num_pending_good_conditions[X] = 0 
        q_good_bfs.append(X)
   
    # Initialize for other nodes k: if P[k]=X or Q[k]=X, one/two conditions are met.
    for k in range(N):
        if k == X:
            continue
        
        original_pending = num_pending_good_conditions[k] # should be 2
        if P[k] == X:
            num_pending_good_conditions[k] -= 1
        if Q[k] == X:
            num_pending_good_conditions[k] -= 1
        
        if num_pending_good_conditions[k] == 0 and original_pending > 0: # Became 0 due to P[k]=X and/or Q[k]=X
             if not is_good[k]:
                is_good[k] = True
                q_good_bfs.append(k)
    
    # BFS to propagate "goodness"
    while q_good_bfs:
        j = q_good_bfs.popleft() # j is now confirmed good
        
        # Consider predecessors k of j via P-edges (P[k] = j)
        for k_prec_P in rev_P_edges[j]:
            if is_good[k_prec_P]: 
                continue 
            num_pending_good_conditions[k_prec_P] -= 1
            if num_pending_good_conditions[k_prec_P] == 0:
                is_good[k_prec_P] = True
                q_good_bfs.append(k_prec_P)
           
        # Consider predecessors k of j via Q-edges (Q[k] = j)
        for k_prec_Q in rev_Q_edges[j]:
            if is_good[k_prec_Q]:
                continue
            num_pending_good_conditions[k_prec_Q] -= 1
            if num_pending_good_conditions[k_prec_Q] == 0:
                is_good[k_prec_Q] = True
                q_good_bfs.append(k_prec_Q)

    # Step 2: Check impossibility
    for k in range(N):
        if (A_balls[k] > 0 or B_balls[k] > 0) and not is_good[k]:
            print(-1)
            return

    # Step 3: Compute minimum operations
    s_ops_member = [False] * N 
    num_ops = 0
    q_ops_bfs = collections.deque()

    for k in range(N):
        if k != X and (A_balls[k] > 0 or B_balls[k] > 0):
            if not s_ops_member[k]: 
                s_ops_member[k] = True
                num_ops += 1
                q_ops_bfs.append(k)
    
    while q_ops_bfs:
        curr = q_ops_bfs.popleft()
        
        next_box_P = P[curr]
        if next_box_P != X and not s_ops_member[next_box_P]:
            s_ops_member[next_box_P] = True
            num_ops += 1
            q_ops_bfs.append(next_box_P)
            
        next_box_Q = Q[curr]
        if next_box_Q != X and not s_ops_member[next_box_Q]:
            s_ops_member[next_box_Q] = True
            num_ops += 1
            q_ops_bfs.append(next_box_Q)
            
    print(num_ops)

solve()