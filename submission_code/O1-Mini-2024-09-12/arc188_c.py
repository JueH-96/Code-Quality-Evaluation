# YOUR CODE HERE
import sys
import sys
import sys
import sys
from collections import defaultdict, deque
import sys
import sys
import sys

def solve():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    graph = [[] for _ in range(N + 1)]
    testimonies = []
    for _ in range(M):
        A, B, C = map(int, sys.stdin.readline().split())
        testimonies.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))
    # Build spanning tree using BFS
    parent = [0] * (N + 1)
    depth = [0] * (N + 1)
    LOG = 20
    ancestors = [[-1] * (N + 1) for _ in range(LOG)]
    is_tree_edge = [False] * M
    used = [False] * (N +1)
    queue = deque()
    queue.append(1)
    used[1] = True
    idx = 0
    # Assign an index to each testimony for later reference
    testimony_idx = defaultdict(list)
    for i, (A,B,C) in enumerate(testimonies):
        testimony_idx[A].append((B, C, i))
        testimony_idx[B].append((A, C, i))
    # Store non-tree edges
    non_tree_edges = []
    # To assign parents and depth
    visited = [False] * (N + 1)
    visited[1] = True
    queue = deque()
    queue.append(1)
    while queue:
        u = queue.popleft()
        for (v, C) in graph[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] +1
                queue.append(v)
            else:
                if v != parent[u]:
                    if u < v:  # To avoid adding both directions
                        non_tree_edges.append((u, v, C))
    # Precompute ancestors for LCA
    for i in range(1, N +1):
        ancestors[0][i] = parent[i]
    for k in range(1, LOG):
        for i in range(1, N +1):
            if ancestors[k-1][i] != -1:
                ancestors[k][i] = ancestors[k-1][ancestors[k-1][i]]
            else:
                ancestors[k][i] = -1
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        for k in reversed(range(LOG)):
            if ancestors[k][u] != -1 and depth[ancestors[k][u]] >= depth[v]:
                u = ancestors[k][u]
        if u == v:
            return u
        for k in reversed(range(LOG)):
            if ancestors[k][u] != -1 and ancestors[k][u] != ancestors[k][v]:
                u = ancestors[k][u]
                v = ancestors[k][v]
        return parent[u]
    # Collect equations
    equations = []
    for (u, v, C_edge) in non_tree_edges:
        l = lca(u, v)
        path = []
        sum_d = C_edge
        temp_u = u
        while temp_u != l:
            path.append(temp_u)
            # Find the testimony from temp_u to parent[temp_u]
            # Assuming unique testimonies as per constraints
            # Need to find the C_Ai for testimony where A_i=temp_u and B_i=parent[temp_u]
            # Since testimonies are unique, iterate through testimonies
            # To speed it up, we can build a map
            # But here, we'll iterate through testimonies to find the correct one
            # It's acceptable as per constraints
            found = False
            for (B, C, idx_t) in testimony_idx[temp_u]:
                if B == parent[temp_u]:
                    sum_d ^= C
                    found = True
                    break
            if not found:
                # Should not happen due to spanning tree
                pass
            temp_u = parent[temp_u]
        temp_v = v
        while temp_v != l:
            path.append(temp_v)
            # Similarly, find the testimony from temp_v to parent[temp_v]
            found = False
            for (B, C, idx_t) in testimony_idx[temp_v]:
                if B == parent[temp_v]:
                    sum_d ^= C
                    found = True
                    break
            if not found:
                pass
            temp_v = parent[temp_v]
        # Now add the current edge's C_j
        path.append(u)
        sum_d ^= C_edge
        # The constraint is sum of C_j for A_i in path equals sum_d
        equation_vars = path.copy()
        equations.append( (set(equation_vars), sum_d) )
    # Now perform Gaussian elimination on the equations
    basis = {}
    for vars_set, rhs in equations:
        vars_list = sorted(vars_set)
        if not vars_list:
            if rhs !=0:
                print(-1)
                return
            else:
                continue
        lead = vars_list[0]
        if lead in basis:
            # XOR the current equation with the basis equation
            existing_vars, existing_rhs = basis[lead]
            new_vars = vars_set.symmetric_difference(existing_vars)
            new_rhs = rhs ^ existing_rhs
            if not new_vars:
                if new_rhs !=0:
                    print(-1)
                    return
                else:
                    continue
            else:
                vars_set = new_vars
                rhs = new_rhs
                vars_list = sorted(vars_set)
                lead = vars_list[0]
                if lead in basis:
                    # Repeat the process
                    continue
                else:
                    basis[lead] = (vars_set, rhs)
        else:
            basis[lead] = (vars_set, rhs)
    # Now, assign C_j variables
    C = [0] * (N +1)
    # Assign variables starting from highest lead
    sorted_leads = sorted(basis.keys(), reverse=True)
    for lead in sorted_leads:
        vars_set, rhs = basis[lead]
        total = rhs
        for var in vars_set:
            if var != lead:
                total ^= C[var]
        C[lead] = total
    # Now, any unset C_j can be 0
    # Build the output string
    output = ''.join(['1' if C[i] else '0' for i in range(1, N +1)])
    print(output)