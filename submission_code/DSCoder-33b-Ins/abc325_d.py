# YOUR CODE HERE

import heapq

n = int(input())
products = []
for _ in range(n):
    t, d = map(int, input().split())
    products.append((t, d))

products.sort()

pq = []
time = 0
count = 0
for t, d in products:
    while pq and pq[0] < t:
        heapq.heappop(pq)
    if not pq or pq[0] > t:
        heapq.heappush(pq, t + d)
        count += 1
    elif pq[0] <= t:
        heapq.heappop(pq)
        heapq.heappush(pq, t + d)

print(count)