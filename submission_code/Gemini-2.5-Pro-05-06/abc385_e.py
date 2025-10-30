import sys

def solve():
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N)]
    deg = [0] * N
    
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    max_S = 0
    if N < 3: # Smallest snowflake has 3 vertices
        # This case should not be reached based on problem constraints (N >= 3)
        # If it were, one might argue 0 snowflake vertices can be formed, N deleted.
        # Or based on problem: "always possible", implies N>=3.
        # The problem constraints N >= 3 make this if-block mostly defensive.
        # A single node or an edge is not a snowflake tree.
        # Smallest snowflake $x=1,y=1$ has $1+1+1*1=3$ vertices.
        # If $N < 3$, this value of $max_S=0$ results in $N-0=N$ deletions.
         pass


    # Iterate over all possible central vertices C
    for C_node_idx in range(N):
        # Neighbors of C are potential arm vertices
        # For each potential arm, calculate its k_val = deg[arm] - 1
        # This is the number of leaves it can support (excluding C itself)
        potential_arm_k_values = [] 
        for neighbor_of_C_idx in adj[C_node_idx]:
            k_val = deg[neighbor_of_C_idx] - 1 
            if k_val >= 1: # Arm must support at least y=1 leaf
                potential_arm_k_values.append(k_val)
        
        if not potential_arm_k_values: # No valid arm candidates for this C
            continue

        # Sort potential arms by k_val descending
        # This allows us to iterate x, and for each x, the y value is determined
        # by the x-th arm in this sorted list (which has the smallest k_val among chosen x arms)
        potential_arm_k_values.sort(reverse=True)
        
        num_potential_arms = len(potential_arm_k_values)
        
        # Iterate through number of arms chosen (x)
        for x in range(1, num_potential_arms + 1):
            # If we choose x arms, they are the ones with the x largest k_values.
            # The effective y will be limited by the arm with the smallest k_value among these x.
            # This is potential_arm_k_values[x-1] due to descending sort.
            y = potential_arm_k_values[x-1]
            
            # y must be at least 1. This is already guaranteed by `if k_val >= 1` check.
            
            current_S = 1 + x + x * y # Total vertices in this snowflake configuration
            if current_S > max_S:
                max_S = current_S
    
    # The problem guarantees N >= 3 and that a snowflake tree can always be formed.
    # The smallest snowflake tree (x=1, y=1) has 3 vertices.
    # So, max_S will be at least 3 if N >= 3.
    # If max_S remained 0 (e.g. if N < 3 and we didn't initialize max_S properly),
    # we could set a default. But given N>=3, it is fine.
    # For N=3, path 0-1-2:
    # C=0, N(0)={1}. deg(1)=2. k_1 = 2-1=1. K_list=[1]. x=1, y=1. S=1+1+1=3. max_S=3.
    # C=1, N(1)={0,2}. deg(0)=1, deg(2)=1. k_0=0, k_2=0. K_list=[]. No score.
    # C=2, N(2)={1}. deg(1)=2. k_1 = 2-1=1. K_list=[1]. x=1, y=1. S=1+1+1=3. max_S=3.
    # N - max_S = 3 - 3 = 0.
    
    print(N - max_S)

solve()