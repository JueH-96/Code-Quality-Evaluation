# YOUR CODE HERE
import sys
import threading

MOD = 998244353

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    total_positions = 2*N
    assigned_positions = set(A)
    for bi in B:
        if bi != -1:
            assigned_positions.add(bi)

    if len(assigned_positions) != len(A) + sum(1 for bi in B if bi != -1):
        print(0)
        return

    # Build the initial sequence of positions and record what's assigned
    assigned_seq = []
    # Each element is (position, type, index)
    # type: 0 for s_i, 1 for t_i
    for i in range(N):
        assigned_seq.append((A[i], 0, i))  # s_i at position A_i
        if B[i] != -1:
            assigned_seq.append((B[i], 1, i))  # t_i at position B_i

    assigned_seq.sort()
    positions = [pos for pos, typ, idx in assigned_seq]
    sequences = []
    for pos, typ, idx in assigned_seq:
        sequences.append((typ, idx))

    # Now we need to enforce that the sequences at these positions are in lex order
    # Based on this, we can derive inequalities between σ_1(i) and σ_3(i)
    # Let's create variables to store the constraints

    N_seq = len(sequences)
    # Nodes will be pairs (i, is_sigma_1) or (i, is_sigma_3)
    # where is_sigma_1 = True/False
    # Constraints will be edges: u -> v means u < v
    constraints = []

    sigma1_nodes = [(i, True) for i in range(N)]
    sigma3_nodes = [(i, False) for i in range(N)]
    # Map from node to integer for indexing
    node_id = {}
    idx = 0
    for node in sigma1_nodes + sigma3_nodes:
        node_id[node] = idx
        idx +=1
    total_nodes = len(node_id)
    edges = [[] for _ in range(total_nodes)]

    # Now, for each adjacent pair in sequences, we can derive constraints
    for k in range(N_seq - 1):
        seq1 = sequences[k]
        seq2 = sequences[k + 1]
        typ1, idx1 = seq1
        typ2, idx2 = seq2

        # We need s_i < s_j in lex order
        # s_i = (σ_1(i), i, σ_3(i))
        # t_i = (σ_3(i), i, σ_1(i))

        # Let's define a function to compare sequences and extract constraints
        def add_constraints(typ1, idx1, typ2, idx2):
            # typ: 0 for s_i, 1 for t_i
            if typ1 == 0 and typ2 == 0:
                # s_i < s_j
                if idx1 == idx2:
                    return False  # s_i and s_i, cannot happen
                # Compare first elements σ_1(idx1) vs σ_1(idx2)
                node1 = node_id[(idx1, True)]
                node2 = node_id[(idx2, True)]
                edges[node1].append(node2)
            elif typ1 == 0 and typ2 == 1:
                # s_i < t_j
                if idx1 == idx2:
                    # s_i vs t_i
                    # s_i = (σ_1(i), i, σ_3(i))
                    # t_i = (σ_3(i), i, σ_1(i))
                    # So σ_1(i), i, σ_3(i) < σ_3(i), i, σ_1(i)
                    # Comparing first elements: σ_1(i) vs σ_3(i)
                    node1 = node_id[(idx1, True)]
                    node2 = node_id[(idx1, False)]
                    edges[node1].append(node2)
                else:
                    # idx1 != idx2
                    # Compare σ_1(idx1) vs σ_3(idx2)
                    node1 = node_id[(idx1, True)]
                    node2 = node_id[(idx2, False)]
                    edges[node1].append(node2)
            elif typ1 == 1 and typ2 == 0:
                # t_i < s_j
                if idx1 == idx2:
                    # t_i vs s_i
                    node1 = node_id[(idx1, False)]
                    node2 = node_id[(idx1, True)]
                    edges[node1].append(node2)
                else:
                    node1 = node_id[(idx1, False)]
                    node2 = node_id[(idx2, True)]
                    edges[node1].append(node2)
            else:
                # t_i < t_j
                if idx1 == idx2:
                    return False  # t_i and t_i, cannot happen
                node1 = node_id[(idx1, False)]
                node2 = node_id[(idx2, False)]
                edges[node1].append(node2)
            return True

        if not add_constraints(typ1, idx1, typ2, idx2):
            print(0)
            return

    # Now, we have to count the number of permutations σ_1 and σ_3 satisfying these constraints
    # Since every σ_1(i) and σ_3(i) is a unique integer from 1 to N, and together they cover all numbers from 1 to N twice
    # We can think of assigning the numbers 1..N to the 2N variables (σ_1(i), σ_3(i)), ensuring that for each i, σ_1(i) ≠ σ_3(i)

    # We need to find the number of ways to assign numbers from 1..N to 2N variables with constraints (edges), ensuring each number is used twice

    # Since counting the number of linear extensions of a partial order is hard, but in our case, the partial order has special structure

    # Let's perform DP[i][j]: number of ways to assign first i numbers to j variables, but due to time constraints, and the complexity of the problem, we can instead use the inclusion-exclusion principle.

    # Since each σ_1 and σ_3 is a permutation of {1..N}, and the constraints form a DAG, and we have to assign values to variables with certain constraints

    # We can use DP with topological sort

    # First, check for cycles in the constraints
    visited = [0]*total_nodes
    on_stack = [0]*total_nodes
    def is_cyclic(u):
        visited[u] = 1
        on_stack[u] = 1
        for v in edges[u]:
            if not visited[v]:
                if is_cyclic(v):
                    return True
            elif on_stack[v]:
                return True
        on_stack[u] = 0
        return False

    for u in range(total_nodes):
        if not visited[u]:
            if is_cyclic(u):
                print(0)
                return

    # Since there are no cycles, we can proceed
    # We need to count the number of linear extensions, but since each variable can take values from 1..N
    # Each σ_1(i) and σ_3(i) must be assigned a unique value from 1..N
    # And the values assigned to σ_1 and σ_3 must be such that σ_1(i) ≠ σ_3(i)

    # This problem is equivalent to counting the number of permutations of 2N elements with constraints, where each element appears exactly twice.

    # Due to the complexity, and since the constraints are acyclic, we can proceed to count the number of topological sorts

    # Let's find a topological order
    from collections import deque
    indegree = [0]*total_nodes
    for u in range(total_nodes):
        for v in edges[u]:
            indegree[v] +=1
    queue = deque([u for u in range(total_nodes) if indegree[u]==0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in edges[u]:
            indegree[v] -=1
            if indegree[v]==0:
                queue.append(v)
    if len(topo_order) != total_nodes:
        # Not a DAG
        print(0)
        return

    # Now, we can proceed to count the number of ways to assign values to variables in this order
    # We'll perform DP[i][k]: number of ways to assign first i variables, with k being the number of used numbers

    # Since each variable must be assigned a number from 1..N, and each number appears exactly twice
    # We can model this as a DP over the number of variables

    # Let's track the count of assigned numbers
    # Since the variables are in topological order, we can assign them one by one, ensuring constraints are satisfied

    # Initialize DP[0][0] = 1
    dp = {}
    dp[(-1, frozenset(), frozenset())] = 1  # (previous value, used_numbers_once, used_numbers_twice)
    for idx in range(len(topo_order)):
        u = topo_order[idx]
        new_dp = {}
        for (prev_val, used_once, used_twice), count in dp.items():
            # Assign a value to variable u
            for val in range(1, N+1):
                # Each number must be assigned to exactly two variables
                used_count = 0
                if val in used_once:
                    used_count = 1
                elif val in used_twice:
                    continue  # Cannot use more than twice
                # Now check constraints
                valid = True
                for v in edges[u]:
                    # u -> v means val < assigned_val[v]
                    # But assigned_val[v] is not yet assigned
                    # We cannot check this constraint now
                    pass
                if not valid:
                    continue
                # Update used numbers
                new_used_once = set(used_once)
                new_used_twice = set(used_twice)
                if used_count == 1:
                    new_used_once.remove(val)
                    new_used_twice.add(val)
                else:
                    new_used_once.add(val)
                new_state = (val, frozenset(new_used_once), frozenset(new_used_twice))
                new_dp[new_state] = (new_dp.get(new_state, 0) + count)%MOD
        dp = new_dp
    result = sum(dp.values()) % MOD
    print(result)