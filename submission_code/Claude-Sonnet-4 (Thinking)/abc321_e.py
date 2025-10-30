def solve(N, X, K):
    if K == 0:
        return 1
    
    count = 0
    
    # Generate all ancestors of X (including X itself)
    ancestors = []
    current = X
    while current >= 1:
        ancestors.append(current)
        if current == 1:
            break
        current = current // 2
    
    # For each possible number of steps up
    for up_steps in range(min(K + 1, len(ancestors))):
        ancestor = ancestors[up_steps]
        down_steps = K - up_steps
        
        if down_steps == 0:
            if up_steps > 0:  # Don't count X when up_steps=0
                count += 1
        else:
            # Count descendants of ancestor at distance down_steps
            total = count_descendants_at_distance(ancestor, down_steps, N)
            
            if up_steps == 0:
                # Going down from X itself
                count += total
            else:
                # Going down from ancestor, but exclude the subtree containing X
                child_towards_x = ancestors[up_steps - 1]
                excluded = count_descendants_at_distance(child_towards_x, down_steps - 1, N)
                count += total - excluded
    
    return count

def count_descendants_at_distance(node, distance, N):
    if distance == 0:
        return 1 if node <= N else 0
    
    start = node * (1 << distance)
    end = start + (1 << distance) - 1
    
    if start > N:
        return 0
    
    return min(end, N) - start + 1

T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(solve(N, X, K))