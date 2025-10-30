N, Q = map(int, input().split())
A = list(map(int, input().split()))

for _ in range(Q):
    b, k = map(int, input().split())
    
    # Calculate distances from point b to all points in A
    distances = []
    for a in A:
        dist = abs(a - b)
        distances.append(dist)
    
    # Sort distances and get kth smallest
    distances.sort()
    print(distances[k-1])