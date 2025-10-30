from collections import deque

def solve():
    H, W = map(int, input().split())
    
    A = []
    for _ in range(H):
        row = list(map(int, input().split()))
        A.append(row)
    
    B = []
    for _ in range(H):
        row = list(map(int, input().split()))
        B.append(row)
    
    # Check if transformation is possible
    # Both grids must have the same multiset of elements
    elements_A = []
    elements_B = []
    
    for i in range(H):
        for j in range(W):
            elements_A.append(A[i][j])
            elements_B.append(B[i][j])
    
    elements_A.sort()
    elements_B.sort()
    
    if elements_A != elements_B:
        print(-1)
        return
    
    # Convert grids to tuples for hashing
    def grid_to_tuple(grid):
        return tuple(tuple(row) for row in grid)
    
    target = grid_to_tuple(B)
    start = grid_to_tuple(A)
    
    if start == target:
        print(0)
        return
    
    # BFS to find minimum operations
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        current_tuple, ops = queue.popleft()
        
        # Convert back to list for manipulation
        current = [list(row) for row in current_tuple]
        
        # Try all possible row swaps (adjacent rows)
        for i in range(H - 1):
            # Swap rows i and i+1
            current[i], current[i+1] = current[i+1], current[i]
            new_state = grid_to_tuple(current)
            
            if new_state == target:
                print(ops + 1)
                return
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, ops + 1))
            
            # Swap back
            current[i], current[i+1] = current[i+1], current[i]
        
        # Try all possible column swaps (adjacent columns)
        for j in range(W - 1):
            # Swap columns j and j+1
            for i in range(H):
                current[i][j], current[i][j+1] = current[i][j+1], current[i][j]
            
            new_state = grid_to_tuple(current)
            
            if new_state == target:
                print(ops + 1)
                return
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, ops + 1))
            
            # Swap back
            for i in range(H):
                current[i][j], current[i][j+1] = current[i][j+1], current[i][j]
    
    print(-1)

solve()