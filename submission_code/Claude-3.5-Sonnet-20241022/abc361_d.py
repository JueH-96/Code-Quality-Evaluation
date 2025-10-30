def can_transform(N, S, T):
    def get_state_str(state):
        return ''.join(state)
    
    def get_next_states(state):
        states = []
        # Find empty cells
        empty1 = state.index('.')
        empty2 = state.index('.', empty1 + 1)
        
        # Try all possible pairs of adjacent stones
        for i in range(N+1):
            if i+1 >= len(state):
                continue
            if state[i] != '.' and state[i+1] != '.' and '.' not in (state[i], state[i+1]):
                new_state = list(state)
                stones = state[i:i+2]
                # Remove stones
                new_state[i] = '.'
                new_state[i+1] = '.'
                # Place stones in empty cells
                new_state[empty1] = stones[0]
                new_state[empty2] = stones[1]
                states.append(get_state_str(new_state))
        return states

    from collections import deque
    
    initial = S + '..'
    target = T + '..'
    
    # BFS
    queue = deque([(initial, 0)])
    seen = {initial}
    
    while queue:
        curr_state, moves = queue.popleft()
        
        if curr_state[:N] == target[:N]:
            return moves
            
        for next_state in get_next_states(list(curr_state)):
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, moves + 1))
                
    return -1

N = int(input())
S = input()
T = input()
print(can_transform(N, S, T))