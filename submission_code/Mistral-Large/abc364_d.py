import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

a = list(map(int, data[2:2+N]))
queries = []

index = 2 + N
for _ in range(Q):
    b = int(data[index])
    k = int(data[index + 1])
    queries.append((b, k))
    index += 2

results = []

for b, k in queries:
    distances = []
    for point in a:
        distance = abs(point - b)
        heapq.heappush(distances, distance)
        if len(distances) > k:
            heapq.heappop(distances)
    results.append(distances[0])

for result in results:
    print(result)