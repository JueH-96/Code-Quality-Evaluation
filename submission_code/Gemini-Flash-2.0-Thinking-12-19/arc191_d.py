# YOUR CODE HERE
import sys
from collections import deque

# Increase recursion depth if needed (not for BFS usually)
# sys.setrecursionlimit(2000)

def solve():
    # Read N, M, S, T
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    S = int(line1[2])
    T = int(line1[3])

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # State is (pos_A, pos_B)
    initial_state = (S, T)
    target_state = (T, S)

    # Distance dictionary: (pos_A, pos_B) -> min_moves
    # Use a dictionary for manageable memory if reachable states are few
    dist = {}
    dist[initial_state] = 0

    # Queue for BFS
    # Store (pos_A, pos_B) tuples
    q = deque([initial_state])

    # BFS loop
    while q:
        u, v = q.popleft()
        d = dist[(u, v)]

        # Check if target state is reached
        if (u, v) == target_state:
            print(d)
            return # Found shortest path

        # Explore neighbors by moving piece A
        for u_prime in adj[u]:
            # Constraint: pieces cannot be on the same vertex
            if u_prime != v:
                next_state = (u_prime, v)
                # If this state hasn't been visited
                if next_state not in dist:
                    dist[next_state] = d + 1
                    q.append(next_state)

        # Explore neighbors by moving piece B
        for v_prime in adj[v]:
            # Constraint: pieces cannot be on the same vertex
            if u != v_prime:
                next_state = (u, v_prime)
                # If this state hasn't been visited
                if next_state not in dist:
                    dist[next_state] = d + 1
                    q.append(next_state)

    # If the queue becomes empty and target state was not reached
    print(-1)

# Call the solve function
solve()