import sys
from collections import deque

def solve():
    """
    Solves the "Boxes and Balls" problem by modeling ball movements as paths on graphs
    and using topological sort to check for dependency cycles.
    """
    # Fast I/O
    readline = sys.stdin.readline
    
    try:
        # 1. Read input and adjust to 0-based indexing
        line = readline()
        if not line: return # Handle empty input for local testing
        N_str, X_str = line.split()
        N, X = int(N_str), int(X_str)
        X -= 1
        
        A = list(map(int, readline().split()))
        B = list(map(int, readline().split()))
        P = [p - 1 for p in map(int, readline().split())]
        Q = [q - 1 for q in map(int, readline().split())]
    except (IOError, ValueError):
        return

    # 2. Precompute inverse permutations for backward traversal
    P_inv = [0] * N
    Q_inv = [0] * N
    for i in range(N):
        P_inv[P[i]] = i
        Q_inv[Q[i]] = i

    # 3. Calculate distances to X in the red graph
    dist_R = [-1] * N
    dist_R[X] = 0
    curr = X
    d = 1
    while True:
        prev = P_inv[curr]
        if prev == X: break
        dist_R[prev] = d
        d += 1
        curr = prev

    # 4. Calculate distances to X in the blue graph
    dist_B = [-1] * N
    dist_B[X] = 0
    curr = X
    d = 1
    while True:
        prev = Q_inv[curr]
        if prev == X: break
        dist_B[prev] = d
        d += 1
        curr = prev

    # 5. Check reachability for all balls and find max distances
    max_dist_R = 0
    max_dist_B = 0
    for i in range(N):
        if A[i] == 1:
            if dist_R[i] == -1:
                print(-1)
                return
            max_dist_R = max(max_dist_R, dist_R[i])
        if B[i] == 1:
            if dist_B[i] == -1:
                print(-1)
                return
            max_dist_B = max(max_dist_B, dist_B[i])

    # 6. Determine the set of nodes that must be operated on (S_op)
    op_nodes_list = []
    is_op_node = [False] * N
    for i in range(N):
        is_on_red_path = (dist_R[i] != -1) and (0 < dist_R[i] <= max_dist_R)
        is_on_blue_path = (dist_B[i] != -1) and (0 < dist_B[i] <= max_dist_B)
        if is_on_red_path or is_on_blue_path:
            op_nodes_list.append(i)
            is_op_node[i] = True
    
    num_op_nodes = len(op_nodes_list)
    if num_op_nodes == 0:
        print(0)
        return

    # 7. Cycle check using topological sort (Kahn's algorithm)
    in_degree = [0] * N
    for u in op_nodes_list:
        is_on_red_path = (dist_R[u] != -1) and (0 < dist_R[u] <= max_dist_R)
        is_on_blue_path = (dist_B[u] != -1) and (0 < dist_B[u] <= max_dist_B)

        if is_on_red_path:
            v = P[u]
            if is_op_node[v]:
                in_degree[v] += 1
        
        if is_on_blue_path:
            v = Q[u]
            if is_op_node[v]:
                in_degree[v] += 1

    queue = deque([u for u in op_nodes_list if in_degree[u] == 0])
    
    processed_count = 0
    while queue:
        u = queue.popleft()
        processed_count += 1
        
        is_on_red_path = (dist_R[u] != -1) and (0 < dist_R[u] <= max_dist_R)
        is_on_blue_path = (dist_B[u] != -1) and (0 < dist_B[u] <= max_dist_B)

        if is_on_red_path:
            v = P[u]
            if is_op_node[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if is_on_blue_path:
            v = Q[u]
            if is_op_node[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    if processed_count == num_op_nodes:
        print(num_op_nodes)
    else:
        print(-1)

if __name__ == "__main__":
    solve()