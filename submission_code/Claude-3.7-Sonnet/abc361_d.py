from collections import deque

def min_operations(n, s, t):
    # Check if the number of black and white stones match
    if s.count('B') != t.count('B') or s.count('W') != t.count('W'):
        return -1
    
    initial_state = s + '..'  # Initial state with the stones in cells 1 to N and cells N+1 and N+2 empty
    target_state = t + '..'   # Target state with the stones in cells 1 to N as per t and cells N+1 and N+2 empty
    
    if initial_state == target_state:
        return 0
    
    # BFS to find the shortest path
    queue = deque([(initial_state, 0)])  # (state, number of operations)
    visited = {initial_state}
    
    while queue:
        state, ops = queue.popleft()
        
        # Find all pairs of adjacent cells that both contain stones
        for i in range(len(state) - 1):
            if state[i] in ['B', 'W'] and state[i+1] in ['B', 'W']:
                # Find the two empty cells
                empty_positions = [j for j in range(len(state)) if state[j] == '.']
                
                # Move the stones from cells i and i+1 to the two empty cells
                new_state = list(state)
                new_state[i] = '.'
                new_state[i+1] = '.'
                new_state[empty_positions[0]] = state[i]
                new_state[empty_positions[1]] = state[i+1]
                new_state = ''.join(new_state)
                
                if new_state == target_state:
                    return ops + 1
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, ops + 1))
    
    return -1  # No solution found

def main():
    n = int(input())
    s = input().strip()
    t = input().strip()
    
    result = min_operations(n, s, t)
    print(result)

if __name__ == "__main__":
    main()