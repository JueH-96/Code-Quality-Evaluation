import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency list for the graph
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed
        adj[u - 1].append(v - 1)
        adj[v - 1].append(u - 1)

    # Track visited vertices for component finding
    visited = [False] * N
    # Store information about each connected component: (size_partition_0, size_partition_1, total_size)
    components_info = [] 

    # Iterate through all vertices
    for i in range(N):
        # If vertex i hasn't been visited, start a new component traversal
        if not visited[i]:
            # BFS queue: stores (vertex, color)
            q = [(i, 0)] 
            visited[i] = True
            
            # Counts for the current component's bipartition sizes
            partition_counts = [0, 0] 
            
            head = 0 # Manual queue head for performance
            while head < len(q):
                u, color = q[head]
                head += 1
                
                # Increment count for the current partition
                partition_counts[color] += 1

                # Explore neighbors
                for v in adj[u]:
                    # If neighbor v hasn't been visited, add to queue with opposite color
                    if not visited[v]:
                        visited[v] = True
                        q.append((v, 1 - color))
                    # Note: Since the problem guarantees the initial graph is bipartite,
                    # we don't need to check for conflicts (neighbor v having same color)
                    # which would indicate an odd cycle.

            # Store component information
            n_i = partition_counts[0] + partition_counts[1]
            l_i = partition_counts[0]
            r_i = partition_counts[1]
            components_info.append((l_i, r_i, n_i))

    # Count initial components with odd size and sum of partition 0 sizes
    num_odd_sized_components = 0
    sum_l_initial = 0

    for l_i, r_i, n_i in components_info:
        if n_i % 2 != 0:
            num_odd_sized_components += 1
        sum_l_initial += l_i # Sum up one side of the partition sizes

    # The game ends when the graph becomes a complete bipartite graph K_{|L|, |R|}.
    # The total number of edges added is |L||R| - M.
    # The winner is determined by the parity of the total number of moves (|L||R| - M).
    # Parity of total moves = (|L||R| + M) mod 2.

    # Case 1: N is odd
    if N % 2 != 0:
        # If N is odd, for any bipartition (L, R) with |L|+|R|=N, |L||R| is always even.
        # This is because if |L| is even, |L||R| is even. If |L| is odd, |R| = N - |L| is odd - odd = even, so |L||R| is even.
        # Total moves parity = (even + M) mod 2 = M mod 2.
        # The winner is determined solely by the parity of M.
        if M % 2 != 0:
            print("Aoki") # Odd moves total, Aoki wins
        else:
            print("Takahashi") # Even moves total, Takahashi wins
            
    # Case 2: N is even
    else:
        # If N is even, |L|*|R| mod 2 = |L| mod 2 * |R| mod 2.
        # Since |L| + |R| = N (even), |L| and |R| must have the same parity.
        # If |L| is even, |R| is even, |L|*|R| is even.
        # If |L| is odd, |R| is odd, |L|*|R| is odd.
        # So, |L|*|R| mod 2 = |L| mod 2.
        # Total moves parity = (|L| + M) mod 2.
        # The winner depends on the parity of |L| and M.

        # The final bipartition (L, R) must be consistent with the initial components' bipartitions.
        # A possible final partition (L, R) is formed by taking a partition (L_i, R_i) for each component C_i
        # and deciding whether to orient it "normally" (L_i -> L, R_i -> R) or "flipped" (R_i -> L, L_i -> R).
        # |L| = sum_{i} size(oriented L_i). size(oriented L_i) is either |L_i| or |R_i|.
        # We showed that the parity of the final |L| can be influenced by the players if and only if
        # there is at least one component with an odd number of vertices initially.
        
        # If there is at least one odd-sized component initially ($k_{init} > 0$),
        # players can force the final |L| to have a specific parity (either even or odd).
        # This is because merging an odd-sized component C_i with any component C_j allows the player
        # to choose the parity of one partition in the resulting component if C_j is also odd-sized.
        # Even if C_j is even-sized, the ability to choose orientations when combining components
        # ensures that if there is at least one odd-sized component, both parities of the final |L| are reachable.
        # Since both parities of |L| are reachable, the first player (Aoki) can choose
        # a sequence of moves that leads to a final state where (|L| + M) is odd, thus winning.
        if num_odd_sized_components > 0:
             print("Aoki")
        # If all components are even-sized initially ($k_{init} = 0$),
        # the parity of the final |L| is fixed regardless of player choices.
        # The final |L| mod 2 is $(\sum l_i) \pmod 2$.
        # Total moves parity = $(\sum l_i + M) \pmod 2$.
        # The winner is determined by this parity.
        else:
            if (sum_l_initial + M) % 2 != 0:
                print("Aoki") # Odd moves total, Aoki wins
            else:
                print("Takahashi") # Even moves total, Takahashi wins

solve()