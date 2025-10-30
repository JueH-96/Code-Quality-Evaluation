def count_subtree_at_depth(root, depth, n):
    """Count nodes at exactly 'depth' distance from 'root' in its subtree"""
    if depth == 0:
        return 1 if root <= n else 0
    
    # Nodes at depth d from root are in range [root * 2^d, (root+1) * 2^d - 1]
    min_val = root * (1 << depth)
    if min_val > n:
        return 0
    
    max_val = (root + 1) * (1 << depth) - 1
    max_val = min(max_val, n)
    
    return max_val - min_val + 1

def solve(n, x, k):
    result = 0
    
    # Build path from x to root
    path = []
    curr = x
    while curr >= 1:
        path.append(curr)
        curr //= 2
    
    # For each node in path
    for i in range(len(path)):
        if i > k:
            break
        
        node = path[i]
        dist_from_x = i
        remaining = k - dist_from_x
        
        if remaining == 0:
            result += 1
        elif remaining > 0:
            if i == 0:
                # At x itself, count all descendants at distance k
                result += count_subtree_at_depth(x, remaining, n)
            else:
                # At ancestor, count descendants excluding the branch containing x
                child_with_x = path[i-1]
                
                # Try left child
                left = 2 * node
                if left <= n and left != child_with_x:
                    result += count_subtree_at_depth(left, remaining - 1, n)
                
                # Try right child
                right = 2 * node + 1
                if right <= n and right != child_with_x:
                    result += count_subtree_at_depth(right, remaining - 1, n)
    
    return result

# Read input
t = int(input())
for _ in range(t):
    n, x, k = map(int, input().split())
    print(solve(n, x, k))