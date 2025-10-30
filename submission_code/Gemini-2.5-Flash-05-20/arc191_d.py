import collections
import sys

def solve():
    # Read N, M, S, T from standard input
    # N: number of vertices
    # M: number of edges
    # S: starting vertex for piece A (1-indexed)
    # T: starting vertex for piece B (1-indexed)
    N, M, S, T = map(int, sys.stdin.readline().split())

    # Convert S and T to 0-indexed for easier list/array access in Python
    S -= 1
    T -= 1

    # Build the adjacency list for the graph
    # adj[i] will contain a list of vertices adjacent to vertex i
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Convert edge vertices to 0-indexed
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u) # Graph is undirected

    # Initialize BFS structures
    # 'visited' dictionary: Stores the minimum number of operations to reach a state (pos_A, pos_B).
    # This also acts as a set to keep track of visited states.
    visited = {}
    
    # 'q' (deque): The BFS queue, storing tuples of (operations_count, pos_A, pos_B).
    q = collections.deque()

    # Add the initial state to the queue
    # Piece A is at S, Piece B is at T, 0 operations so far.
    initial_state = (S, T)
    visited[initial_state] = 0
    q.append((0, S, T))

    # Perform Breadth-First Search
    while q:
        k, u, v = q.popleft() # Dequeue the current state: k=operations, u=A's position, v=B's position

        # Check if the current state is the goal state
        # Goal: A is at T, B is at S
        if u == T and v == S:
            print(k) # If yes, k is the minimum operations, print it and exit
            return

        # --- Explore possible moves for Piece A ---
        # Iterate over all neighbors 'next_u' of A's current position 'u'
        for next_u in adj[u]:
            # Apply the constraint: Piece A cannot move to the vertex currently occupied by Piece B
            if next_u != v:
                # Form the new state after A moves
                next_state = (next_u, v)
                # If this new state has not been visited yet (or found with a higher cost, though BFS guarantees first visit is optimal)
                if next_state not in visited:
                    visited[next_state] = k + 1 # Record the operations count
                    q.append((k + 1, next_u, v)) # Enqueue the new state

        # --- Explore possible moves for Piece B ---
        # Iterate over all neighbors 'next_v' of B's current position 'v'
        for next_v in adj[v]:
            # Apply the constraint: Piece B cannot move to the vertex currently occupied by Piece A
            if next_v != u:
                # Form the new state after B moves
                next_state = (u, next_v)
                # If this new state has not been visited yet
                if next_state not in visited:
                    visited[next_state] = k + 1 # Record the operations count
                    q.append((k + 1, u, next_v)) # Enqueue the new state

    # If the BFS completes (queue becomes empty) and the goal state was never reached,
    # it means it's impossible to achieve the goal.
    print(-1)

# Call the solve function to execute the program
solve()