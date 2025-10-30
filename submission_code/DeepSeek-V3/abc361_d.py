from collections import deque

def min_operations(N, S, T):
    if sorted(S) != sorted(T):
        return -1
    initial = tuple(S) + ('E', 'E')
    target = tuple(T) + ('E', 'E')
    visited = set()
    queue = deque()
    queue.append((initial, 0))
    visited.add(initial)
    while queue:
        current, steps = queue.popleft()
        if current == target:
            return steps
        empty_indices = [i for i, x in enumerate(current) if x == 'E']
        if len(empty_indices) != 2:
            continue
        k1, k2 = empty_indices
        for x in range(N+1):
            if current[x] != 'E' and current[x+1] != 'E':
                new_state = list(current)
                new_state[k1] = new_state[x]
                new_state[k2] = new_state[x+1]
                new_state[x] = 'E'
                new_state[x+1] = 'E'
                new_state_tuple = tuple(new_state)
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    queue.append((new_state_tuple, steps + 1))
    return -1

# Read input
N = int(input())
S = input().strip()
T = input().strip()

# Compute and print the result
result = min_operations(N, S, T)
print(result)