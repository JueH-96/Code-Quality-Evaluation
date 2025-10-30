import collections
import sys

def solve():
    N, M, S, T = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Compute shortest distances from S and T using BFS
    # This helps determine the parity difference for state representation
    def bfs_distances(start_node):
        dist = [-1] * (N + 1)
        q = collections.deque([start_node])
        dist[start_node] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    dist_S = bfs_distances(S)
    dist_T = bfs_distances(T)

    # State for BFS: (pos_A, pos_B, delta_A, delta_B)
    # pos_A: current vertex of piece A
    # pos_B: current vertex of piece B
    # delta_A = (k_A - dist_S[pos_A]) % 2
    # delta_B = (k_B - dist_T[pos_B]) % 2
    # k_A is the number of moves A has made
    # k_B is the number of moves B has made
    # Store minimum total moves k = k_A + k_B to reach this state

    # Use a dictionary to store minimum distances to states, because N is large
    # Key: (pos_A, pos_B, delta_A, delta_B)
    # Value: min_k (minimum total moves)
    dist_states = {}
    q = collections.deque()

    # Initial state: A at S (0 moves), B at T (0 moves)
    # Graph is connected, so dist_S and dist_T are always >= 0
    initial_delta_A = (0 - dist_S[S]) % 2
    initial_delta_B = (0 - dist_T[T]) % 2

    initial_state = (S, T, initial_delta_A, initial_delta_B)
    dist_states[initial_state] = 0
    q.append(initial_state)

    # BFS to find the shortest path in the state space graph
    while q:
        u, v, delta_A, delta_B = q.popleft()
        k = dist_states[(u, v, delta_A, delta_B)]

        # Move A
        for next_u in adj[u]:
            if next_u != v: # Conflict constraint: A cannot move to where B is
                next_k = k + 1

                # Calculate new delta_A and delta_B
                # k_A increases by 1, k_B stays the same
                # new_delta_A = (k_A_old + 1 - dist_S[next_u]) % 2
                # k_A_old = dist_S[u] + delta_A + 2 * i
                # k_A_old + 1 = dist_S[u] + delta_A + 1 + 2 * i
                # new_delta_A = (dist_S[u] + delta_A + 1 - dist_S[next_u]) % 2
                new_delta_A = (dist_S[u] + delta_A + 1 - dist_S[next_u]) % 2
                new_delta_B = delta_B # k_B doesn't change, so delta_B relative to dist_T[v] doesn't change

                next_state = (next_u, v, new_delta_A, new_delta_B)
                if next_state not in dist_states:
                    dist_states[next_state] = next_k
                    q.append(next_state)

        # Move B
        for next_v in adj[v]:
            if next_v != u: # Conflict constraint: B cannot move to where A is
                next_k = k + 1

                # Calculate new delta_A and delta_B
                # k_B increases by 1, k_A stays the same
                # new_delta_B = (k_B_old + 1 - dist_T[next_v]) % 2
                # k_B_old = dist_T[v] + delta_B + 2 * j
                # k_B_old + 1 = dist_T[v] + delta_B + 1 + 2 * j
                # new_delta_B = (dist_T[v] + delta_B + 1 - dist_T[next_v]) % 2
                new_delta_A = delta_A # k_A doesn't change, so delta_A relative to dist_S[u] doesn't change
                new_delta_B = (dist_T[v] + delta_B + 1 - dist_T[next_v]) % 2

                next_state = (u, next_v, new_delta_A, new_delta_B)
                if next_state not in dist_states:
                    dist_states[next_state] = next_k
                    q.append(next_state)

    # After BFS, find the minimum distance to any target state (T, S, delta_A, delta_B)
    # The goal is to reach pos_A = T and pos_B = S. The parities (delta_A, delta_B) can be anything.
    # We need the minimum k among all reachable states (T, S, d_A, d_B) for d_A, d_B in {0, 1}.
    min_dist_target = -1
    for d_A in range(2):
        for d_B in range(2):
            target_state = (T, S, d_A, d_B)
            if target_state in dist_states:
                if min_dist_target == -1 or dist_states[target_state] < min_dist_target:
                    min_dist_target = dist_states[target_state]

    sys.stdout.write(str(min_dist_target) + '
')

solve()