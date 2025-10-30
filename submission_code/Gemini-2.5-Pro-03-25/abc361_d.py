# YOUR CODE HERE
import collections
import sys

# Function to solve the problem
def solve():
    # Read input: N - number of stones, S - initial configuration, T - target configuration
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Basic check: If the number of white ('W') and black ('B') stones
    # differ between S and T, it's impossible to reach the target configuration.
    # Comparing sorted strings is an easy way to check if the multisets of characters are identical.
    if sorted(S) != sorted(T):
        print("-1")
        return

    # The state of the system includes the N stones and 2 empty cells in N+2 positions.
    # Represent the state as a tuple of characters. Tuples are immutable and hashable,
    # suitable for use in sets (for visited tracking) and queues.
    # Initial state: Stones from S in positions 1 to N, empty cells '.' in positions N+1, N+2.
    # Using 0-based indexing, this corresponds to indices 0 to N-1 for stones, N and N+1 for empty cells.
    initial_state_list = list(S) + ['.', '.']
    initial_state_tuple = tuple(initial_state_list)
    
    # Target state: Stones from T in positions 1 to N, empty cells '.' in positions N+1, N+2.
    target_state_list = list(T) + ['.', '.']
    target_state_tuple = tuple(target_state_list)

    # If the initial state is already the target state, 0 operations are needed.
    if initial_state_tuple == target_state_tuple:
        print(0)
        return

    # Use Breadth-First Search (BFS) to find the minimum number of operations.
    # Initialize a queue with the starting state and its distance (0 operations).
    # collections.deque is efficient for queue operations (append and popleft).
    q = collections.deque([(initial_state_tuple, 0)])
    
    # Keep track of visited states to avoid cycles and redundant computations.
    # Use a set for efficient membership checking (average O(1) time).
    visited = {initial_state_tuple}

    # Start the BFS process
    while q:
        # Dequeue the state to explore: current state tuple and its distance from start.
        curr_state_tuple, dist = q.popleft()

        # Find the index of the first empty cell '.'.
        # The two empty cells are always adjacent. Finding the first one is sufficient.
        # We iterate through the tuple to find the first occurrence of '.'.
        k = -1 # Initialize k, index of the first empty cell (0-based)
        for i in range(N + 2):
            if curr_state_tuple[i] == '.':
                k = i
                break
        # The second empty cell is at index k+1.

        # Explore all possible moves from the current state.
        # A move consists of choosing an adjacent pair of cells (x, x+1) containing stones.
        # The index 'x' here is 0-based, corresponding to the first cell of the pair.
        # The loop range ensures we check pairs (0,1) up to (N, N+1), 
        # corresponding to 1-based cell indices (1,2) up to (N+1, N+2).
        for x in range(N + 1): 
            
            # Check if both cells at indices x and x+1 contain stones (i.e., are not '.')
            if curr_state_tuple[x] != '.' and curr_state_tuple[x+1] != '.':
                
                # If they contain stones, perform the operation:
                # Move the stones from cells x, x+1 to the empty cells k, k+1, preserving order.
                
                # Convert the immutable tuple state to a mutable list to perform modifications.
                next_state_list = list(curr_state_tuple)
                
                # Get the stones from positions x and x+1.
                stone1 = next_state_list[x]
                stone2 = next_state_list[x+1]
                
                # Place the stones into the empty slots k and k+1. Stone from x goes to k, stone from x+1 goes to k+1.
                next_state_list[k] = stone1
                next_state_list[k+1] = stone2
                
                # The original slots x and x+1 now become empty.
                next_state_list[x] = '.'
                next_state_list[x+1] = '.'
                
                # Convert the modified list back to an immutable tuple.
                next_state_tuple = tuple(next_state_list)

                # Check if the resulting state is the target state.
                if next_state_tuple == target_state_tuple:
                    # If yes, we found the shortest path. The distance is dist + 1.
                    print(dist + 1)
                    return # Terminate the function.

                # If the resulting state has not been visited before:
                if next_state_tuple not in visited:
                    # Mark it as visited to avoid re-exploring.
                    visited.add(next_state_tuple)
                    # Add it to the queue for further exploration, with distance incremented by 1.
                    q.append((next_state_tuple, dist + 1))

    # If the queue becomes empty and the target state was not reached,
    # it means the target state is unreachable from the initial state.
    print("-1")

# Call the solve function to run the program logic.
solve()