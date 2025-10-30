# Read the number of points and queries
N, Q = map(int, input().split())

# Read the coordinates of the N points
a = list(map(int, input().split()))

# Process each query
for _ in range(Q):
    # Read the query point b_j and k_j
    b, k = map(int, input().split())
    
    # Calculate the distance from each point to b_j
    distances = [abs(a_i - b) for a_i in a]
    
    # Sort the distances in ascending order
    distances.sort()
    
    # The k-th closest point has the (k-1)th distance in the sorted array
    # (0-indexed)
    print(distances[k-1])