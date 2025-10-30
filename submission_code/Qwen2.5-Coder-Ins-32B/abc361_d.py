# YOUR CODE HERE
from collections import deque

def min_operations(N, S, T):
    def is_valid(state):
        return state.count('.') == 2 and state[:N] == T

    def get_neighbors(state):
        neighbors = []
        empty_cells = [i for i, c in enumerate(state) if c == '.']
        if len(empty_cells) != 2:
            return neighbors
        k, k1 = empty_cells
        if k1 != k + 1:
            return neighbors
        for x in range(k - 1, -1, -1):
            if state[x] != '.' and state[x + 1] != '.':
                new_state = list(state)
                new_state[x], new_state[x + 1] = '.', '.'
                new_state[k], new_state[k1] = state[x], state[x + 1]
                neighbors.append(''.join(new_state))
        return neighbors

    queue = deque([(S, 0)])
    visited = set([S])

    while queue:
        current_state, operations = queue.popleft()
        if is_valid(current_state):
            return operations
        for neighbor in get_neighbors(current_state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, operations + 1))

    return -1

import sys
input = sys.stdin.read().split()
N = int(input[0])
S = input[1]
T = input[2]

print(min_operations(N, S, T))