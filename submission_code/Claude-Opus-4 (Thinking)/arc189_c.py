from collections import deque

N, X = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Convert to 0-indexed
X -= 1
P = [p - 1 for p in P]
Q = [q - 1 for q in Q]

# Initial state: tuple of (red_balls, blue_balls) for each box
initial = tuple((A[i], B[i]) for i in range(N))

# Calculate total balls
total_red = sum(A)
total_blue = sum(B)

# Target state: all balls in box X, other boxes empty
target = tuple((0, 0) if i != X else (total_red, total_blue) for i in range(N))

if initial == target:
    print(0)
else:
    # BFS to find minimum operations
    queue = deque([(initial, 0)])
    visited = {initial}
    ans = -1
    
    while queue:
        state, ops = queue.popleft()
        
        # Try operating on each box except X
        for i in range(N):
            if i == X:  # Cannot operate on target box
                continue
            
            red, blue = state[i]
            if red == 0 and blue == 0:  # Box is empty
                continue
            
            # Create new state after operation
            new_state = list(state)
            new_state[i] = (0, 0)  # Box i becomes empty
            new_state[P[i]] = (new_state[P[i]][0] + red, new_state[P[i]][1])  # Red balls go to P[i]
            new_state[Q[i]] = (new_state[Q[i]][0], new_state[Q[i]][1] + blue)  # Blue balls go to Q[i]
            new_state = tuple(new_state)
            
            if new_state == target:
                ans = ops + 1
                break
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, ops + 1))
        
        if ans != -1:
            break
    
    print(ans)