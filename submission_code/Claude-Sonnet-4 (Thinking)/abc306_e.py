import heapq

N, K, Q = map(int, input().split())
A = [0] * N

for _ in range(Q):
    X, Y = map(int, input().split())
    A[X-1] = Y  # Convert to 0-indexed
    
    # Get K largest elements
    largest_K = heapq.nlargest(K, A)
    result = sum(largest_K)
    print(result)