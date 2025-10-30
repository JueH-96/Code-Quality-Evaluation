import heapq

N, K = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Sort B in descending order for easier access to large values
B_sorted = sorted(B, reverse=True)

# Precompute all (i,k) pairs with their base values and sums
pairs = []
for i in range(N):
    for k in range(N):
        base = A[i] * C[k]
        sum_ac = A[i] + C[k]
        pairs.append((base, sum_ac))

# Sort pairs by sum_ac in descending order to prioritize larger potential values
pairs.sort(key=lambda x: x[1], reverse=True)

# Use a max-heap (negate values for min-heap in Python)
# Each element: (-value, pair_idx, b_idx)
heap = []

# Initialize heap with the largest possible values
# For each pair, start with the largest B value
for pair_idx in range(len(pairs)):
    base, sum_ac = pairs[pair_idx]
    value = B_sorted[0] * sum_ac + base
    heapq.heappush(heap, (-value, pair_idx, 0))

# Extract K largest values
result = []
seen = set()

while len(result) < K and heap:
    neg_value, pair_idx, b_idx = heapq.heappop(heap)
    value = -neg_value
    
    # Add to result
    result.append(value)
    
    # If there's a next B value for this pair, add it to heap
    if b_idx + 1 < N:
        base, sum_ac = pairs[pair_idx]
        next_value = B_sorted[b_idx + 1] * sum_ac + base
        heapq.heappush(heap, (-next_value, pair_idx, b_idx + 1))

print(result[K-1])