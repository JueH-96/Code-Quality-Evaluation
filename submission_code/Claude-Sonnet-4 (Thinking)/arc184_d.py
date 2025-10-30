def solve():
    n = int(input())
    balls = []
    for i in range(n):
        x, y = map(int, input().split())
        balls.append((x, y))
    
    # BFS to find all reachable states
    from collections import deque
    
    initial_state = frozenset(range(n))
    visited = set()
    queue = deque([initial_state])
    visited.add(initial_state)
    
    while queue:
        current_state = queue.popleft()
        
        # Try choosing each ball k in the current state
        for k in current_state:
            x_k, y_k = balls[k]
            new_state = {k}  # Always keep ball k
            
            for i in current_state:
                if i != k:
                    x_i, y_i = balls[i]
                    # Keep ball i if (X_i > X_k AND Y_i < Y_k) OR (X_i < X_k AND Y_i > Y_k)
                    if (x_i > x_k and y_i < y_k) or (x_i < x_k and y_i > y_k):
                        new_state.add(i)
            
            new_state = frozenset(new_state)
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)
    
    print(len(visited) % 998244353)

solve()