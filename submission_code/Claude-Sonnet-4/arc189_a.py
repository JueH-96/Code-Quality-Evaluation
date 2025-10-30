def solve():
    n = int(input())
    target = list(map(int, input().split()))
    
    # Initial configuration
    initial = [i % 2 for i in range(1, n + 1)]
    
    MOD = 998244353
    
    # Use BFS-like approach to count all possible sequences
    from collections import defaultdict, deque
    
    # Count ways to reach each state
    ways = defaultdict(int)
    ways[tuple(initial)] = 1
    
    queue = deque([initial])
    visited = set([tuple(initial)])
    
    while queue:
        current_state = queue.popleft()
        current_tuple = tuple(current_state)
        current_ways = ways[current_tuple]
        
        # Try all possible operations from current state
        for l in range(n):
            for r in range(l + 2, n):
                # Check if operation is valid
                if current_state[l] == current_state[r]:
                    # Check if all intermediate values are different from current_state[l]
                    valid = True
                    for i in range(l + 1, r):
                        if current_state[i] == current_state[l]:
                            valid = False
                            break
                    
                    if valid:
                        # Apply operation
                        new_state = current_state[:]
                        for i in range(l + 1, r):
                            new_state[i] = current_state[l]
                        
                        new_tuple = tuple(new_state)
                        ways[new_tuple] = (ways[new_tuple] + current_ways) % MOD
                        
                        if new_tuple not in visited:
                            visited.add(new_tuple)
                            queue.append(new_state)
    
    return ways[tuple(target)]

print(solve())