import heapq
from collections import defaultdict

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

# Calculate smallest modulus values for all A[i] + 0*C, A[i] + 1*C, ..., A[i] + (M-1)*C
# This is done by checking all patterns for each A[i], but we only keep
# track of the smallest M values since these are the possible "distances"
# from an A[i] to the nearest multiple of M.
patterns = defaultdict(list)
for Ai in A:
    for j in range(M):
        patterns[(Ai + j*C) % M].append(j)

min_values = []
for m in range(M):
    if m in patterns:
        heap = [(d, i) for i, d in enumerate(patterns[m]) if d is not None]
        heapq.heapify(heap)
        min_values.append(heap)

# For each element in the cycle,
# we find the smallest value among all moduli.
# We go k cycles.
answer = 0
for _ in range(K):
   heap = min_values[0]
    best, i = heapq.heappop(heap)
    new_value = best + M
    if len(patterns[best%M]) - 1 > i:
        heapq.heappush(heap, (patterns[best%M][i + 1], i + 1))
    else:
        heapq.heappush(heap, (new_value, 0))
    min_values[0] = heap
    answer += best

print(answer)