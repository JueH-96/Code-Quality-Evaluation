# YOUR CODE HERE
N, Q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(Q):
    b, k = map(int, input().split())
    
    # Calculate distances from all A_i to B
    distances = []
    for i in range(N):
        dist = abs(a[i] - b)
        distances.append(dist)
    
    # Sort distances in ascending order
    distances.sort()
    
    # Output the k-th smallest distance (k is 1-indexed)
    print(distances[k-1])