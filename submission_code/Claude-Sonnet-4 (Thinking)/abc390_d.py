def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    visited = set()
    xor_values = set()
    
    def dfs(state):
        state_tuple = tuple(state)
        if state_tuple in visited:
            return
        visited.add(state_tuple)
        
        # Calculate XOR of current state
        xor_val = 0
        for x in state:
            xor_val ^= x
        xor_values.add(xor_val)
        
        # Try all possible moves: from bag i to bag j
        for i in range(N):
            if state[i] > 0:  # Only move if bag i has stones
                for j in range(N):
                    if i != j:
                        new_state = list(state)
                        new_state[j] += new_state[i]  # Add all stones to bag j
                        new_state[i] = 0              # Empty bag i
                        dfs(new_state)
    
    dfs(A)
    return len(xor_values)

print(solve())