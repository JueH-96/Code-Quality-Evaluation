# YOUR CODE HERE
import collections
import sys

def solve():
    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Initial state representation: tuple of characters
    # N stones from S followed by 2 empty cells '.'
    initial_state = tuple(list(S) + ['.', '.'])

    # Target state representation
    # N stones from T followed by 2 empty cells '.'
    target_state = tuple(list(T) + ['.', '.'])

    # Use BFS to find the minimum number of operations
    # Queue stores tuples of (current_state, distance)
    q = collections.deque([(initial_state, 0)])

    # Set to keep track of visited states to avoid cycles and redundant computations
    visited = {initial_state}

    # BFS loop
    while q:
        # Get the state and distance from the front of the queue
        current_state, d = q.popleft()

        # If the current state is the target state, we have found the shortest path
        if current_state == target_state:
            print(d)
            return

        # Find the 0-indexed position of the first of the two adjacent empty cells
        empty_k = -1
        # The empty pair can start at index 0 up to N (inclusive)
        for i in range(N + 1):
            if current_state[i] == '.' and current_state[i+1] == '.':
                empty_k = i
                break

        # Iterate through all possible pairs of adjacent cells (x, x+1) that contain stones.
        # These are the stone pairs that can be moved.
        # The 0-indexed positions are x and x+1.
        # The problem states 1 <= x <= N+1 for the cell numbers.
        # This corresponds to 0-indexed x ranging from 0 to N.
        # Example: N=6, cells 1..8. x can be 1..7. Pairs (1,2), (2,3), ..., (7,8).
        # Stones are initially in 1..N. Empty at N+1, N+2.
        # A pair of cells (x, x+1) must contain stones.
        # If stones have moved, they can be anywhere except the empty spots.
        # So, iterate through all possible 0-indexed start positions x from 0 to N.
        # Check if cells x and x+1 contain stones.
        for x in range(N + 1):
            # Check if both cells at 0-indexed positions x and x+1 contain stones (i.e., not '.')
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Perform the move operation: move stones from (x, x+1) to (empty_k, empty_k+1)

                # Create a mutable list copy of the current state
                next_state_list = list(current_state)

                # Store the stones being moved (order is preserved)
                stone1 = next_state_list[x]
                stone2 = next_state_list[x+1]

                # Set the source cells to empty
                next_state_list[x] = '.'
                next_state_list[x+1] = '.'

                # Place the stones in the empty cells (empty_k, empty_k+1), preserving their order
                next_state_list[empty_k] = stone1
                next_state_list[empty_k+1] = stone2

                # Convert the list back to an immutable tuple for use in the visited set
                next_state_tuple = tuple(next_state_list)

                # If the resulting state has not been visited before, add it to the queue and visited set
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    q.append((next_state_tuple, d + 1))

    # If the BFS completes and the target state was not reached, it is impossible
    print(-1)

# Execute the solve function
solve()