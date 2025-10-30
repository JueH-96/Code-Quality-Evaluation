from collections import deque

N = int(input())
S = input().strip()
T = input().strip()

# Check if transformation is possible
if sorted(S) != sorted(T):
    print(-1)
else:
    # Initial state
    initial = S + ".."
    target = T + ".."
    
    # BFS
    queue = deque([(initial, 0)])
    visited = {initial}
    
    found = False
    while queue:
        state, steps = queue.popleft()
        
        if state == target:
            print(steps)
            found = True
            break
        
        # Find the first empty cell
        empty_start = state.index('.')
        
        # Try all possible moves
        for i in range(N+1):
            # Check if positions i and i+1 both contain stones
            if state[i] != '.' and state[i+1] != '.':
                # Move stones at positions i and i+1 to empty positions
                new_state = list(state)
                new_state[empty_start] = state[i]
                new_state[empty_start+1] = state[i+1]
                new_state[i] = '.'
                new_state[i+1] = '.'
                new_state = ''.join(new_state)
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
    
    if not found:
        print(-1)