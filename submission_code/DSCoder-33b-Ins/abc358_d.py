# YOUR CODE HERE

import heapq

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

boxes = sorted([(a, i) for i, a in enumerate(A)])

heap = []
j = 0
total_cost = 0
for i in range(M):
    while j < N and boxes[j][0] <= B[i]:
        heapq.heappush(heap, -boxes[j][0])
        j += 1
    if heap:
        total_cost -= heapq.heappop(heap)
    else:
        print(-1)
        exit()

print(total_cost)