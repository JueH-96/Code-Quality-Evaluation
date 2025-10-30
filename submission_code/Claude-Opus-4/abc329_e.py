from collections import deque

def solve():
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    
    # Start with all '#' characters
    initial = '#' * N
    
    # BFS to find if we can reach S from initial state
    queue = deque([initial])
    visited = {initial}
    
    while queue:
        current = queue.popleft()
        
        # Check if we reached the target
        if current == S:
            print("Yes")
            return
        
        # Try all possible positions to place T
        for i in range(N - M + 1):
            # Create new state by replacing M characters starting at position i with T
            new_state = current[:i] + T + current[i + M:]
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)
    
    print("No")

solve()