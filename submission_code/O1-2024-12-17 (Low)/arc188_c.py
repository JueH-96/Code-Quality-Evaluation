def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    testimonies = input_data[2:]
    
    # If there are no testimonies, we can safely say nobody is confused (all zero).
    if M == 0:
        print("0" * N)
        return
    
    # T(a) will hold all (b, c) for testimonies "a -> b, c"
    # We only need to store them grouped by speaker a.
    from collections import defaultdict
    T = defaultdict(list)
    
    idx = 0
    for _ in range(M):
        A = int(testimonies[idx]); B = int(testimonies[idx+1]); C = int(testimonies[idx+2])
        idx += 3
        T[A].append((B, C))
    
    # We will build an undirected graph among villagers (1..N) based on the following idea:
    # For each speaker A who has testimonies (B1, c1), (B2, c2), ..., (Bk, ck),
    # we pick one of them, say (B1, c1), as a "reference".
    # For each other (Bi, ci) with i=2..k, we add an undirected edge (B1, Bi)
    # with XOR label = c1 ^ ci.
    #
    # Why does this work?
    # From the speaker A's perspective, for any two testimonies (B_i, c_i) and (B_j, c_j),
    # we must have y_Bi xor y_Bj = c_i xor c_j (independent of A's confusion).
    # By always pairing each Bi with the reference B1, we can ensure that
    # y_Bi xor y_B1 = c_i xor c_1, which enforces all pairwise constraints transitively.
    
    adj = [[] for _ in range(N+1)]  # adjacency list for villagers 1..N
    # Each entry adj[u] will be a list of (v, label) meaning
    # y_v must be y_u ^ label in any consistent assignment.
    
    # Build edges
    for a in range(1, N+1):
        if len(T[a]) > 1:
            # pick the first one as reference
            ref_b, ref_c = T[a][0]
            # connect each of the remaining to the reference
            for i in range(1, len(T[a])):
                b_i, c_i = T[a][i]
                xor_label = ref_c ^ c_i
                adj[ref_b].append((b_i, xor_label))
                adj[b_i].append((ref_b, xor_label))
        # If T[a] has 0 or 1 testimonies, it does not force any additional "target-to-target" edges.
    
    # Now we check if the "target-to-target" constraints (the edges we just built)
    # can be satisfied with some assignment of y_1..y_N in {0,1}.
    # We'll do this by a standard BFS or DFS for XOR constraints.
    y = [-1]*(N+1)  # -1 means unassigned, else 0 or 1
    
    from collections import deque
    
    for start in range(1, N+1):
        if y[start] == -1 and adj[start]:
            # BFS from start
            y[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for (v, lbl) in adj[u]:
                    if y[v] == -1:
                        y[v] = y[u] ^ lbl
                        queue.append(v)
                    else:
                        # check consistency
                        if y[v] != (y[u] ^ lbl):
                            # conflict => no solution
                            print(-1)
                            return
    
    # For any node still unassigned (and with no edges), we can simply set y[node] = 0
    # because it has no constraints to violate.
    for i in range(1, N+1):
        if y[i] == -1:
            y[i] = 0
    
    # If we reach here, we have a consistent assignment y_1..y_N
    # We must now compute x_1..x_N (confusion statuses).
    # For each villager A:
    #  - if they have at least one testimony, pick that reference (B, c) again
    #    then from the rule y_B xor y_A = c xor x_A  =>  x_A = y_B xor y_A xor c.
    #  - if they have no testimonies, we can pick x_A = 0 arbitrarily.
    
    x = [0]*(N+1)
    for a in range(1, N+1):
        if T[a]:
            # use the first testimony as reference
            b0, c0 = T[a][0]
            # x_A = y_b0 xor y_a xor c0
            x[a] = y[b0] ^ y[a] ^ c0
        else:
            x[a] = 0
    
    # Output the confusion array x[1..N]
    print("".join(str(x[i]) for i in range(1, N+1)))