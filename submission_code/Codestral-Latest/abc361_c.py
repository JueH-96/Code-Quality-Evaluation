import sys
from heapq import heappop, heappush

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Create a min-heap and a max-heap
min_heap = []
max_heap = []

# Initialize the heaps with the first K elements
for i in range(K):
    heappush(min_heap, A[i])
    heappush(max_heap, -A[i])

# Initialize the result with the difference between the maximum and minimum of the first K elements
result = -max_heap[0] - min_heap[0]

# Iterate through the remaining elements
for i in range(K, N):
    # If the current element is smaller than the maximum of the first K elements
    if A[i] < -max_heap[0]:
        heappush(max_heap, -A[i])
        heappop(max_heap)
    # If the current element is larger than the minimum of the first K elements
    if A[i] > min_heap[0]:
        heappush(min_heap, A[i])
        heappop(min_heap)
    # Update the result
    result = min(result, -max_heap[0] - min_heap[0])

print(result)