from collections import deque

def solve():
    N = int(input())
    S = input().strip()
    T = input().strip()
    
    # Check if transformation is possible - same number of B's and W's
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        print(-1)
        return
    
    # Represent state as a string with '.' for empty cells
    initial_state = S + '..'
    target_state = T + '..'
    
    # BFS to find minimum operations
    queue = deque([(initial_state, 0)])
    visited = {initial_state}
    
    while queue:
        state, ops = queue.popleft()
        
        if state == target_state:
            print(ops)
            return
        
        # Find empty cells
        empty_pos = []
        for i in range(N + 2):
            if state[i] == '.':
                empty_pos.append(i)
        
        # Try all possible moves
        for x in range(N + 1):
            # Check if cells x and x+1 both contain stones
            if state[x] != '.' and state[x+1] != '.':
                # Create new state by moving stones from x, x+1 to empty positions
                new_state = list(state)
                
                # Move stones to empty positions (preserving order)
                new_state[empty_pos[0]] = state[x]
                new_state[empty_pos[1]] = state[x+1]
                new_state[x] = '.'
                new_state[x+1] = '.'
                
                new_state_str = ''.join(new_state)
                
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state_str, ops + 1))
    
    print(-1)

solve()