from collections import deque

def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Convert to 0-indexed
    X -= 1
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Find cycle containing a given node in a permutation
    def find_cycle(perm, start):
        cycle = []
        current = start
        while True:
            cycle.append(current)
            current = perm[current]
            if current == start:
                break
        return set(cycle)
    
    red_cycle = find_cycle(P, X)
    blue_cycle = find_cycle(Q, X)
    
    # Check if all red balls are in the red cycle
    for i in range(N):
        if A[i] > 0 and i not in red_cycle:
            return -1
    
    # Check if all blue balls are in the blue cycle
    for i in range(N):
        if B[i] > 0 and i not in blue_cycle:
            return -1
    
    # Use BFS to find minimum operations
    initial_state = tuple(A + B)
    
    # Goal state: all balls in box X
    goal_state = [0] * (2 * N)
    goal_state[X] = sum(A)
    goal_state[X + N] = sum(B)
    goal_state = tuple(goal_state)
    
    if initial_state == goal_state:
        return 0
    
    queue = deque([(initial_state, 0)])
    visited = {initial_state}
    
    while queue:
        state, ops = queue.popleft()
        
        # Try operating on each box
        for i in range(N):
            if state[i] == 0 and state[i + N] == 0:
                continue  # No balls in box i
            
            # Perform operation on box i
            new_state = list(state)
            red_balls = new_state[i]
            blue_balls = new_state[i + N]
            
            # Remove balls from box i
            new_state[i] = 0
            new_state[i + N] = 0
            
            # Add balls to destination boxes
            new_state[P[i]] += red_balls
            new_state[Q[i]] += blue_balls
            
            new_state = tuple(new_state)
            
            if new_state == goal_state:
                return ops + 1
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, ops + 1))
    
    return -1

print(solve())