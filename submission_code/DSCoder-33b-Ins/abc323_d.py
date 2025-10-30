# YOUR CODE HERE
import heapq

N = int(input())
slimes = []
for _ in range(N):
    size, count = map(int, input().split())
    slimes.append((-size, count))

heapq.heapify(slimes)

while len(slimes) > 1:
    size1, count1 = heapq.heappop(slimes)
    size2, count2 = slimes[0]
    if count1 + count2 > 1:
        if count1 > 1:
            heapq.heapreplace(slimes, (size1, count1 - 1))
        else:
            heapq.heappop(slimes)
        heapq.heappush(slimes, (size2 * 2, 1))

print(len(slimes) + slimes[0][1])