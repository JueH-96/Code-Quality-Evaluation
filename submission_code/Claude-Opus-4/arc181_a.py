def min_operations(n, p):
    # Check if already sorted
    if all(p[i] == i + 1 for i in range(n)):
        return 0
    
    # Try all possible sequences of operations using BFS
    from collections import deque
    
    # Convert to tuple for hashing
    initial = tuple(p)
    target = tuple(range(1, n + 1))
    
    if initial == target:
        return 0
    
    visited = {initial}
    queue = deque([(initial, 0)])
    
    while queue:
        current, ops = queue.popleft()
        
        # Try all possible k values
        for k in range(1, n + 1):
            # Create new permutation after operation
            new_perm = list(current)
            
            # Sort left part if k >= 2
            if k >= 2:
                left_part = sorted(new_perm[:k-1])
                new_perm[:k-1] = left_part
            
            # Sort right part if k <= n-1
            if k <= n - 1:
                right_part = sorted(new_perm[k:])
                new_perm[k:] = right_part
            
            new_tuple = tuple(new_perm)
            
            if new_tuple == target:
                return ops + 1
            
            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append((new_tuple, ops + 1))
    
    return -1  # Should never reach here

# Read input
t = int(input())
for _ in range(t):
    n = int(input())
    p = list(map(int, input().split()))
    print(min_operations(n, p))