from collections import deque

def solve():
    N = int(input())
    S = input().strip()
    T = input().strip()
    
    # Check if transformation is possible
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        return -1
    
    # If already equal
    if S == T:
        return 0
    
    # BFS to find minimum operations
    initial_state = S + '..'
    target_state = T + '..'
    
    queue = deque([(initial_state, 0)])
    visited = {initial_state}
    
    while queue:
        current_state, ops = queue.popleft()
        
        # Find empty positions
        empty_pos = []
        for i in range(N + 2):
            if current_state[i] == '.':
                empty_pos.append(i)
        
        # Try all possible moves
        for x in range(N + 1):
            # Check if positions x and x+1 both have stones
            if current_state[x] != '.' and current_state[x + 1] != '.':
                # Create new state by moving stones from x, x+1 to empty positions
                new_state = list(current_state)
                
                # Move stones to empty positions
                new_state[empty_pos[0]] = current_state[x]
                new_state[empty_pos[1]] = current_state[x + 1]
                new_state[x] = '.'
                new_state[x + 1] = '.'
                
                new_state_str = ''.join(new_state)
                
                # Check if we reached target
                if new_state_str == target_state:
                    return ops + 1
                
                # Add to queue if not visited
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state_str, ops + 1))
    
    return -1

print(solve())