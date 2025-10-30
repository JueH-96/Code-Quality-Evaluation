from collections import deque

def min_operations(N, S, T):
    # Convert input strings to lists of characters
    S_list = list(S)
    T_list = list(T)

    # Check if the desired state is already achieved
    if S_list == T_list:
        return 0

    # Initialize a queue for the BFS
    queue = deque([(S_list, 0)])

    while queue:
        current_state, operations = queue.popleft()

        # Check if the desired state is achieved
        if current_state == T_list:
            return operations

        # Try all possible moves
        for i in range(N):
            if i < N - 1 and current_state[i] != '.' and current_state[i + 1] != '.':
                # Move the stones from cells i and i+1 to the empty cells
                new_state = current_state[:]
                new_state[i], new_state[i + 1] = '.', '.'
                new_state[N], new_state[N + 1] = current_state[i], current_state[i + 1]
                queue.append((new_state, operations + 1))

    # If the desired state cannot be achieved, return -1
    return -1

# Read the input from stdin
N = int(input())
S = input()
T = input()

# Solve the problem and print the answer
print(min_operations(N, S, T))