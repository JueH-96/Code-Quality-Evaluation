import heapq

N, K = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Sort in descending order
A.sort(reverse=True)
B.sort(reverse=True)
C.sort(reverse=True)

# Max heap (use negative values for max heap with heapq)
heap = []
visited = set()

# Compute the value for a given (i, j, k)
def compute_value(i, j, k):
    return A[i] * B[j] + B[j] * C[k] + C[k] * A[i]

# Start with the maximum possible value
value = compute_value(0, 0, 0)
heapq.heappush(heap, (-value, 0, 0, 0))
visited.add((0, 0, 0))

for count in range(K):
    neg_value, i, j, k = heapq.heappop(heap)
    value = -neg_value
    
    if count == K - 1:
        print(value)
        break
    
    # Add neighbors
    neighbors = [(i+1, j, k), (i, j+1, k), (i, j, k+1)]
    for ni, nj, nk in neighbors:
        if ni < N and nj < N and nk < N and (ni, nj, nk) not in visited:
            new_value = compute_value(ni, nj, nk)
            heapq.heappush(heap, (-new_value, ni, nj, nk))
            visited.add((ni, nj, nk))