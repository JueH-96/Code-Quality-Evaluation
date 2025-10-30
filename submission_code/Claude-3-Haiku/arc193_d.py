from collections import deque

def solve_case(N, A, B):
    # Initialize the positions of the pieces
    positions = [i for i in range(N) if A[i] == '1']
    
    # Check if the initial configuration satisfies the condition
    if all(B[i] == '1' for i in positions) and all(B[i] == '0' for i in range(N) if i not in positions):
        return 0
    
    # Initialize the queue for BFS
    queue = deque([(positions, 0)])
    visited = set()
    
    while queue:
        curr_positions, steps = queue.popleft()
        
        # Check if the current configuration satisfies the condition
        if all(B[i] == '1' for i in curr_positions) and all(B[i] == '0' for i in range(N) if i not in curr_positions):
            return steps
        
        # Mark the current configuration as visited
        visited.add(tuple(curr_positions))
        
        # Generate the next configurations by moving the pieces
        for i in range(1, N+1):
            new_positions = [pos - 1 if pos < i else pos + 1 for pos in curr_positions]
            if tuple(new_positions) not in visited:
                queue.append((new_positions, steps + 1))
    
    # If no configuration satisfies the condition, return -1
    return -1

# Read the input
T = int(input())
for _ in range(T):
    N = int(input())
    A = input()
    B = input()
    print(solve_case(N, A, B))