def min_operations_to_transform(N, S, T):
    # Count the number of white and black stones in S and T
    count_S = {'W': S.count('W'), 'B': S.count('B')}
    count_T = {'W': T.count('W'), 'B': T.count('B')}
    
    # If the counts of stones do not match, return -1
    if count_S != count_T:
        return -1
    
    # Function to calculate the minimum number of moves
    def bfs(start, target):
        from collections import deque
        
        queue = deque([(start, 0)])  # (current state, number of moves)
        visited = set()
        visited.add(start)
        
        while queue:
            current, moves = queue.popleft()
            
            if current == target:
                return moves
            
            # Try to move adjacent pairs of stones
            for i in range(N - 1):
                if current[i] != '.' and current[i + 1] != '.':
                    # Create a new state by moving stones
                    new_state = list(current)
                    new_state[i], new_state[i + 1] = '.', '.'
                    new_state = ''.join(new_state)
                    
                    # Add the two stones to the empty cells
                    if '.' in new_state:
                        empty_index = new_state.index('.')
                        if empty_index + 1 < N + 2 and new_state[empty_index + 1] == '.':
                            new_state = list(new_state)
                            new_state[empty_index], new_state[empty_index + 1] = current[i], current[i + 1]
                            new_state = ''.join(new_state)
                            
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append((new_state, moves + 1))
        
        return -1
    
    # Initial state and target state
    initial_state = S + '..'
    target_state = T + '..'
    
    return bfs(initial_state, target_state)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]
T = data[2]

# Get the result
result = min_operations_to_transform(N, S, T)

# Print the result
print(result)