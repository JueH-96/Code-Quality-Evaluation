import collections
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    ops_raw = []
    for _ in range(M):
        ops_raw.append(list(map(int, sys.stdin.readline().split())))

    # Store operations for efficient lookup
    # op1_starts_at[u]: list of (R_k, k) for operations k where L_k - 1 == u
    op1_starts_at = collections.defaultdict(list)
    # op2_triggers[u]: list of (R_k, k) for operations k where L_k - 1 == u
    # This represents the Op2 choice for (L_k, R_k), which means:
    # if prefix [1, L_k-1] (i.e., [1, u]) is covered, we can pay 1 to cover the rest ([R_k+1, N]).
    # So, we can transition from u to N with cost 1.
    op2_triggers = collections.defaultdict(list)

    for i in range(M):
        L, R = ops_raw[i]
        op1_starts_at[L - 1].append((R, i))
        op2_triggers[L - 1].append((R, i)) 

    # dist[j] = minimum cost to cover positions [1, j]
    dist = [float('inf')] * (N + 1)
    
    # parent_data[j] = (previous_node, operation_index, operation_type) for path reconstruction
    # operation_type: 0 for implicit (u->u+1), 1 for Op1, 2 for Op2
    # operation_index: original index of operation (0 to M-1), or -1 for implicit
    parent_data = [None] * (N + 1)

    dist[0] = 0
    q = collections.deque()
    q.appendleft(0) # Store only the node 'u', cost is dist[u]

    while q:
        u = q.popleft()
        current_cost = dist[u]

        # Optimization: if a shorter path to u has already been found and processed, skip
        # This can happen if an item is added to both ends of deque
        if u > 0 and parent_data[u] is not None and current_cost > dist[u]:
             continue # This line is theoretically not needed for 0-1 BFS if not adding to both ends, but harmless.

        # Rule 1: Implicit edge (u, u+1) with cost 0
        if u + 1 <= N:
            if dist[u+1] > current_cost:
                dist[u+1] = current_cost
                parent_data[u+1] = (u, -1, 0) # (prev_u, op_idx, op_type)
                q.appendleft(u+1)

        # Rule 2: Op1 (L_k-1 -> R_k) with cost 1
        # For all operations that START at u+1 (i.e., L_k-1 == u)
        for R_k, op_idx in op1_starts_at[u]:
            if dist[R_k] > current_cost + 1:
                dist[R_k] = current_cost + 1
                parent_data[R_k] = (u, op_idx, 1)
                q.append(R_k) # Cost 1, so append to right

        # Rule 3: Op2 (L_k-1 -> N) with cost 1
        # For all operations where L_k-1 == u
        for R_k, op_idx in op2_triggers[u]: # R_k not strictly used here, but (L_k, R_k) defines the operation
            if dist[N] > current_cost + 1:
                dist[N] = current_cost + 1
                parent_data[N] = (u, op_idx, 2)
                q.append(N) # Cost 1, so append to right

    if dist[N] == float('inf'):
        sys.stdout.write("-1
")
    else:
        sys.stdout.write(str(dist[N]) + "
")
        
        # Reconstruct the chosen operations
        ans_ops = [0] * M
        curr = N
        
        while curr > 0:
            prev_u, op_idx, op_type = parent_data[curr]
            
            if op_type == 0: # Implicit (u -> u+1) edge, no operation used for this step
                curr = prev_u
            elif op_type == 1: # Operation Type 1 was used for op_idx
                ans_ops[op_idx] = 1
                curr = prev_u
            elif op_type == 2: # Operation Type 2 was used for op_idx
                ans_ops[op_idx] = 2
                # This operation covered [1, prev_u] and [R_k+1, N].
                # We reached N from prev_u by using this operation.
                # So the next node to backtrack is prev_u.
                curr = prev_u
                
        sys.stdout.write(*(map(str, ans_ops)))
        sys.stdout.write("
")

# Call the solver function
solve()