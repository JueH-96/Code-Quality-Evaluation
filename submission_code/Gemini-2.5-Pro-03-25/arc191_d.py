# YOUR CODE HERE
import sys
import collections

# Setting a higher recursion depth is generally not necessary for iterative BFS, 
# but can be included as a precaution for potentially complex graph structures 
# if any recursive helper functions were used (which they are not here).
# sys.setrecursionlimit(200000 + 10) 

def solve():
    """
    Solves the problem using Breadth-First Search (BFS) on the state space.
    The state is defined by the positions of piece A and piece B.
    Finds the minimum number of moves to swap pieces A and B from initial vertices S and T
    to target vertices T and S, respectively, avoiding collisions.
    """
    
    # Read input values: N vertices, M edges, start vertex S for piece A, start vertex T for piece B.
    N, M, S, T = map(int, sys.stdin.readline().split())
    
    # Adjust vertex numbers to be 0-based for internal processing (Python uses 0-based indexing).
    S -= 1 
    T -= 1 
    
    # Build adjacency list representation of the graph.
    # collections.defaultdict(list) automatically handles initialization of lists for new keys.
    adj = collections.defaultdict(list)
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Adjust edge vertices to 0-based indexing.
        # Add edge in both directions since the graph is undirected.
        adj[u-1].append(v-1)
        adj[v-1].append(u-1)

    # Initialize the BFS queue.
    # Each element in the queue is a tuple: ((pos_A, pos_B), moves)
    # The state (pos_A, pos_B) is represented as a tuple, which is hashable and can be used in sets/dictionaries.
    # Start with the initial state (S, T) and 0 moves.
    q = collections.deque([((S, T), 0)]) 
    
    # Keep track of visited states to avoid cycles and redundant computations.
    # Using a set provides efficient average time complexity O(1) for checking membership (visited.add and 'in' check).
    # Stores tuples (pos_A, pos_B) representing visited states.
    visited = set() 
    visited.add((S, T)) # Mark the initial state as visited.
    
    # Start the BFS process. The loop continues as long as there are states to explore in the queue.
    while q:
        # Dequeue the current state and its associated cost (number of moves `k`) from the front of the queue.
        current_state_tuple, k = q.popleft()
        curr_A, curr_B = current_state_tuple
        
        # --- Try moving piece A ---
        # Iterate through all neighbors of piece A's current vertex `curr_A`.
        for next_A in adj[curr_A]:
            # Check if this potential move reaches the target state (A at T, B at S).
            if next_A == T and curr_B == S:
                 print(k + 1) # Print the total number of moves (k + 1 for this final move).
                 return # Shortest path found, exit the function.

            # Check if the move is valid according to the rules:
            # Piece A cannot move to the vertex currently occupied by piece B (`curr_B`).
            if next_A != curr_B: 
                # Define the tuple representing the next state after moving A.
                next_state = (next_A, curr_B)
                # Check if this state has been visited before.
                if next_state not in visited: 
                    # If not visited, mark it as visited.
                    visited.add(next_state)
                    # Enqueue the new state and its cost (k + 1 moves).
                    q.append((next_state, k + 1))

        # --- Try moving piece B ---
        # Iterate through all neighbors of piece B's current vertex `curr_B`.
        for next_B in adj[curr_B]:
             # Check if this potential move reaches the target state (A at T, B at S).
             if curr_A == T and next_B == S:
                 print(k + 1) # Print the total number of moves (k + 1 for this final move).
                 return # Shortest path found, exit the function.

             # Check if the move is valid according to the rules:
             # Piece B cannot move to the vertex currently occupied by piece A (`curr_A`).
             if curr_A != next_B: 
                # Define the tuple representing the next state after moving B.
                next_state = (curr_A, next_B)
                 # Check if this state has been visited before.
                if next_state not in visited: 
                    # If not visited, mark it as visited.
                    visited.add(next_state)
                     # Enqueue the new state and its cost (k + 1 moves).
                    q.append((next_state, k + 1))
                    
    # If the BFS queue becomes empty and the target state was never reached,
    # it means the target state is unreachable from the initial state under the given rules.
    print(-1)

# Execute the solve function to run the program according to the problem specification.
solve()