from collections import deque

def min_operations(N, S, T):
    # Check if it's possible to achieve the desired state
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        return -1

    # Initialize the queue with the initial state
    queue = deque([(S, 0)])

    # Initialize a set to keep track of visited states
    visited = {S}

    while queue:
        state, steps = queue.popleft()

        # Check if the current state is the desired state
        if state == T:
            return steps

        # Generate all possible next states
        for i in range(N - 1):
            # Check if the current cell and the next cell are not empty
            if state[i] != '.' and state[i + 1] != '.':
                # Find the first empty cell
                for j in range(N + 1, -1, -1):
                    if state[j] == '.':
                        break

                # Create a new state by moving the stones
                new_state = list(state)
                new_state[i] = '.'
                new_state[i + 1] = '.'
                new_state[j] = state[i]
                new_state[j + 1] = state[i + 1]
                new_state = ''.join(new_state)

                # Check if the new state has not been visited before
                if new_state not in visited:
                    queue.append((new_state, steps + 1))
                    visited.add(new_state)

    # If it's impossible to achieve the desired state, return -1
    return -1

# Read the inputs from stdin
N = int(input())
S = input()
T = input()

# Solve the problem and write the answer to stdout
print(min_operations(N, S, T))