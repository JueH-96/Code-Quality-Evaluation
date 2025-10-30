import collections

def solve():
    N = int(input())
    S_str = input()
    T_str = input()

    # 1. Check if S and T have the same counts of 'W' and 'B'.
    # If not, it's impossible to transform S into T, as stone colors do not change.
    counts_S = collections.Counter(S_str)
    counts_T = collections.Counter(T_str)
    if counts_S['W'] != counts_T['W'] or counts_S['B'] != counts_T['B']:
        print(-1)
        return

    # Initial state representation: tuple of N+2 characters.
    # Cells are 0-indexed: 0 to N+1.
    # Initially, stones are in cells 0 to N-1, and cells N, N+1 are empty.
    initial_state_list = list(S_str) + ['.', '.']
    initial_state_tuple = tuple(initial_state_list)

    # Target state representation: tuple of N+2 characters.
    # The problem specifies that in the target state, cells 1 to N contain stones
    # matching T_i, which implies cells N+1 and N+2 (0-indexed N and N+1) are empty.
    target_state_list = list(T_str) + ['.', '.']
    target_state_tuple = tuple(target_state_list)

    # BFS queue: stores (current_state_tuple, operations_count)
    q = collections.deque()
    # Visited set: stores current_state_tuple to avoid redundant computations and cycles
    visited = set()

    # Start BFS from the initial state
    q.append((initial_state_tuple, 0))
    visited.add(initial_state_tuple)

    while q:
        current_state, operations = q.popleft()

        # If current_state matches the target state, we found the minimum operations
        if current_state == target_state_tuple:
            print(operations)
            return

        # Find the indices of the two empty cells in the current state
        empty_indices = []
        for i in range(N + 2): # Iterate through all N+2 cells (0 to N+1)
            if current_state[i] == '.':
                empty_indices.append(i)
        # Assuming there are always exactly two empty cells
        k1, k2 = empty_indices[0], empty_indices[1]

        # Iterate through all possible pairs of adjacent cells (x, x+1)
        # 'x' is the 0-indexed position of the first cell in the pair.
        # The problem states 1 <= x <= N+1 (1-indexed).
        # In 0-indexed, this means x can range from 0 to N.
        # For x=N, the pair is (N, N+1), which are the last two cells.
        for x in range(N + 1):
            # Check if both cells x and x+1 contain stones (i.e., are not empty)
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Create a mutable list from the current state tuple to modify it
                next_state_list = list(current_state)

                # Get the colors of the stones to be moved
                stone1_color = next_state_list[x]
                stone2_color = next_state_list[x+1]

                # Move stones from (x, x+1) to (k1, k2), preserving their order
                next_state_list[k1] = stone1_color
                next_state_list[k2] = stone2_color

                # Mark the original positions (x, x+1) as empty
                next_state_list[x] = '.'
                next_state_list[x+1] = '.'

                # Convert the list back to a tuple for hashing and storage in the set/queue
                next_state_tuple = tuple(next_state_list)

                # If this new state has not been visited, add it to the queue
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    q.append((next_state_tuple, operations + 1))

    # If the queue becomes empty and the target state was never reached, it's impossible
    print(-1)

# Execute the solver function
solve()