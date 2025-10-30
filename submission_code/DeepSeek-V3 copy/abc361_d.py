from collections import deque

def min_operations(N, S, T):
    # Initialize the initial state
    initial_state = list(S) + ['.', '.']
    target_state = list(T) + ['.', '.']
    
    # Convert states to tuples for hashing
    initial_tuple = tuple(initial_state)
    target_tuple = tuple(target_state)
    
    if initial_tuple == target_tuple:
        return 0
    
    # BFS queue: (state, number of operations)
    queue = deque()
    queue.append((initial_tuple, 0))
    
    # Visited set to keep track of visited states
    visited = set()
    visited.add(initial_tuple)
    
    while queue:
        current_state, ops = queue.popleft()
        
        # Find the positions of the empty cells
        empty_indices = [i for i, x in enumerate(current_state) if x == '.']
        k1, k2 = empty_indices
        
        # Iterate over all possible pairs of adjacent stones
        for x in range(N+1):
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Create a new state by moving the stones to the empty cells
                new_state = list(current_state)
                new_state[k1] = new_state[x]
                new_state[k2] = new_state[x+1]
                new_state[x] = '.'
                new_state[x+1] = '.'
                new_tuple = tuple(new_state)
                
                if new_tuple == target_tuple:
                    return ops + 1
                
                if new_tuple not in visited:
                    visited.add(new_tuple)
                    queue.append((new_tuple, ops + 1))
    
    return -1

# Read input
N = int(input())
S = input().strip()
T = input().strip()

# Compute and print the result
result = min_operations(N, S, T)
print(result)