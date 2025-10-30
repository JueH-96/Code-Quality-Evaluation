import sys
from collections import deque

def solve():
    # Read N, M, S, T
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    S = int(line1[2]) - 1 # 0-indexed
    T = int(line1[3]) - 1 # 0-indexed

    # Read edges and build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj[u].append(v)
        adj[v].append(u)

    # BFS on the state space (pos_A, pos_B)
    # A state is defined by the positions of piece A and piece B.
    # We use a dictionary to store the minimum number of moves to reach each state (u, v).
    # The dictionary also implicitly serves as a visited set.
    dist = {}

    # Initial state: Piece A is at S, Piece B is at T. Distance is 0.
    initial_state = (S, T)
    dist[initial_state] = 0
    
    # Queue for BFS. Stores states (pos_A, pos_B).
    q = deque([initial_state])

    # Target state: Piece A is at T, Piece B is at S.
    target_state = (T, S)

    while q:
        # Dequeue the current state
        u, v = q.popleft()
        k = dist[(u, v)] # Get the minimum moves to reach this state

        # If the current state is the target state, we found the shortest path.
        if u == target_state[0] and v == target_state[1]:
            print(k)
            return # Exit after printing the result

        next_k = k + 1

        # Explore possible next states by moving one piece

        # Move piece A
        # Piece A is currently at vertex u. It can move to any adjacent vertex u_prime.
        for u_prime in adj[u]:
            # A move is valid if the destination vertex u_prime is not occupied by piece B (vertex v).
            if u_prime != v:
                next_state = (u_prime, v)
                # If this state has not been visited yet (distance is not in dist dictionary)
                if next_state not in dist:
                    # The distance to the next state is one more than the current state's distance.
                    dist[next_state] = next_k
                    # Enqueue the next state for exploration.
                    q.append(next_state)

        # Move piece B
        # Piece B is currently at vertex v. It can move to any adjacent vertex v_prime.
        for v_prime in adj[v]:
            # A move is valid if the destination vertex v_prime is not occupied by piece A (vertex u).
            if v_prime != u:
                next_state = (u, v_prime)
                # If this state has not been visited yet
                if next_state not in dist:
                    # The distance to the next state is one more than the current state's distance.
                    dist[next_state] = next_k
                    # Enqueue the next state for exploration.
                    q.append(next_state)

    # If the queue becomes empty and the target state was not reached, it's impossible.
    print(-1)

solve()