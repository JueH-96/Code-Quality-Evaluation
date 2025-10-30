def solve():
    n = int(input())
    x = list(map(int, input().split()))
    
    # Try all possible operations and find minimum sum
    min_sum = sum(x)
    
    # We need to explore all possible states reachable by operations
    # Since operations preserve sum of 4 consecutive elements,
    # we need to find the configuration that minimizes total sum
    
    # Use BFS to explore all reachable states
    from collections import deque
    
    visited = set()
    queue = deque([tuple(x)])
    visited.add(tuple(x))
    
    while queue:
        current = list(queue.popleft())
        current_sum = sum(current)
        min_sum = min(min_sum, current_sum)
        
        # Try all possible operations
        for i in range(n - 3):
            # Make a copy for the operation
            new_state = current[:]
            
            # Find positions in sorted order
            positions = [(new_state[j], j) for j in range(n)]
            positions.sort()
            
            # Find the i-th and (i+3)-rd pieces in ascending order
            pos_i = positions[i][1]
            pos_i3 = positions[i+3][1]
            pos_i1 = positions[i+1][1]
            pos_i2 = positions[i+2][1]
            
            # Calculate midpoint
            m = (new_state[pos_i] + new_state[pos_i3]) / 2
            
            # Move the (i+1)-th and (i+2)-th pieces
            new_state[pos_i1] = 2 * m - new_state[pos_i1]
            new_state[pos_i2] = 2 * m - new_state[pos_i2]
            
            # Check if this state was visited
            state_tuple = tuple(sorted(new_state))
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append(new_state)
    
    print(int(min_sum))

solve()