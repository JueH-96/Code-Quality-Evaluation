import sys
from collections import deque

def solve():
    N = int(input())
    S = input().strip()
    T = input().strip()
    
    # Convert strings to tuples of 0s and 1s for easier manipulation
    S = tuple(1 if c == 'B' else 0 for c in S)
    T = tuple(1 if c == 'B' else 0 for c in T)
    
    # Initialize the queue for BFS and the visited set
    queue = deque([(S + (0, 0), 0)])
    visited = set([S + (0, 0)])
    
    while queue:
        current, steps = queue.popleft()
        if current[:N] == T:
            print(steps)
            return
        
        # Try moving stones from each pair of adjacent cells
        for i in range(N + 1):
            if current[i] != 2 and current[i + 1] != 2:
                new_state = list(current)
                new_state[i], new_state[i + 1] = 2, 2  # Mark the cells as empty
                new_state[-2], new_state[-1] = current[i], current[i + 1]  # Move stones to the last two cells
                new_state = tuple(new_state)
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
                    
    print(-1)

solve()