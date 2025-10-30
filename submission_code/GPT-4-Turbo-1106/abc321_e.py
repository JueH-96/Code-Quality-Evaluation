import sys

def count_vertices_at_distance(N, X, K):
    # If K is 0, the answer is always 1, since the distance from X to itself is 0.
    if K == 0:
        return 1
    
    # Calculate the depth of X
    depth_X = X.bit_length() - 1
    
    # If K is greater than the depth of X, no vertices can be at that distance.
    if K > depth_X:
        return 0
    
    # Calculate the maximum possible depth in the tree
    max_depth = N.bit_length()
    
    # If the distance K is within the depth of X, calculate the number of vertices
    if depth_X >= K:
        # Calculate the number of vertices at depth K below X
        vertices_below = 1 << K
        
        # Calculate the depth of the lowest possible vertex at distance K from X
        lowest_depth = depth_X + K
        
        # If the lowest possible vertex at distance K is beyond the maximum depth, adjust the count
        if lowest_depth > max_depth or (1 << lowest_depth) > N:
            vertices_below -= ((1 << lowest_depth) - N - 1) // 2
        
        # Calculate the number of vertices at depth K above X
        vertices_above = 0
        if K <= depth_X:
            vertices_above = 1
        
        return vertices_below + vertices_above
    else:
        return 0

# Read the number of test cases
T = int(sys.stdin.readline().strip())

# Process each test case
for _ in range(T):
    N, X, K = map(int, sys.stdin.readline().strip().split())
    print(count_vertices_at_distance(N, X, K))