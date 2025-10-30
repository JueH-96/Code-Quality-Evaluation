# YOUR CODE HERE
N, Q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(Q):
    b, k = map(int, input().split())
    
    # Calculate distances from all A points to current B point
    distances = []
    for ai in a:
        distances.append(abs(ai - b))
    
    # Sort distances in ascending order
    distances.sort()
    
    # Get the k-th smallest distance (k is 1-indexed)
    result = distances[k - 1]
    print(result)