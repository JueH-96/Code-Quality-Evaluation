def solve(N, X, K):
    if K == 0:
        return 1
    
    count = 0
    
    # For each possible number of steps going up
    for up in range(K + 1):
        down = K - up
        
        # Find the ancestor after going up 'up' steps
        ancestor = X
        for _ in range(up):
            if ancestor == 1:
                ancestor = 0  # Above root
                break
            ancestor //= 2
        
        if ancestor == 0:
            continue
            
        if down == 0:
            # We stop at the ancestor
            if up > 0:  # Don't count X itself when up=0
                count += 1
            continue
        
        # Count nodes at distance 'down' from ancestor
        # But exclude the subtree containing X
        nodes_at_distance = count_at_distance(ancestor, down, N)
        
        if up > 0:
            # Subtract nodes in the subtree that contains X
            x_ancestor = X
            for _ in range(up - 1):
                x_ancestor //= 2
            nodes_in_x_subtree = count_at_distance(x_ancestor, down - 1, N)
            nodes_at_distance -= nodes_in_x_subtree
        
        count += nodes_at_distance
    
    return count

def count_at_distance(root, distance, N):
    if distance == 0:
        return 1 if root <= N else 0
    
    # Nodes at distance 'distance' from root are in range
    # [root * 2^distance, (root+1) * 2^distance - 1]
    left = root * (1 << distance)
    right = (root + 1) * (1 << distance) - 1
    
    if left > N:
        return 0
    
    return min(right, N) - left + 1

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(solve(N, X, K))