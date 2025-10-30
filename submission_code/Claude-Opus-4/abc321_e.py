def count_vertices_at_distance(N, X, K):
    if K == 0:
        return 1
    
    # Find ancestors of X up to distance K
    ancestors = []
    curr = X
    dist = 0
    while curr >= 1 and dist <= K:
        ancestors.append((curr, dist))
        if curr == 1:
            break
        curr = curr // 2
        dist += 1
    
    total = 0
    
    # For each ancestor, count vertices at the required distance
    for vertex, dist_from_x in ancestors:
        remaining_dist = K - dist_from_x
        
        if remaining_dist == 0:
            total += 1
        else:
            # Count descendants at distance remaining_dist from vertex
            # but exclude the direction we came from
            if dist_from_x == 0:
                # We're at X itself, count all descendants at distance K
                total += count_descendants_at_distance(N, vertex, remaining_dist)
            else:
                # We're at an ancestor of X
                # Count descendants at distance remaining_dist, but exclude the subtree containing X
                child_to_exclude = ancestors[dist_from_x - 1][0]
                total += count_descendants_at_distance_excluding(N, vertex, remaining_dist, child_to_exclude)
    
    return total

def count_descendants_at_distance(N, vertex, dist):
    if dist == 0:
        return 1 if vertex <= N else 0
    
    # Count vertices at exactly distance 'dist' from 'vertex'
    # These are at depth = depth(vertex) + dist
    min_val = vertex * (2 ** dist)
    max_val = vertex * (2 ** dist) + (2 ** dist) - 1
    
    if min_val > N:
        return 0
    
    return min(max_val, N) - min_val + 1

def count_descendants_at_distance_excluding(N, vertex, dist, exclude_child):
    if dist == 0:
        return 1 if vertex <= N else 0
    
    total = 0
    
    # Consider left child
    left_child = 2 * vertex
    if left_child <= N and left_child != exclude_child:
        total += count_descendants_at_distance(N, left_child, dist - 1)
    elif left_child == exclude_child:
        # Need to go deeper but exclude this path
        total += count_descendants_at_distance_excluding(N, left_child, dist - 1, exclude_child)
    
    # Consider right child
    right_child = 2 * vertex + 1
    if right_child <= N and right_child != exclude_child:
        total += count_descendants_at_distance(N, right_child, dist - 1)
    elif right_child == exclude_child:
        # Need to go deeper but exclude this path
        total += count_descendants_at_distance_excluding(N, right_child, dist - 1, exclude_child)
    
    return total

# Read input
T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    print(count_vertices_at_distance(N, X, K))