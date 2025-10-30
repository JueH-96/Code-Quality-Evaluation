import heapq

n = int(input())
count_map = {}
heap = []

for _ in range(n):
    s, c = map(int, input().split())
    count_map[s] = c
    heapq.heappush(heap, s)

total_pairs = 0

while heap:
    x = heapq.heappop(heap)
    if x not in count_map or count_map[x] == 0:
        continue
    current = count_map[x]
    pairs = current // 2
    total_pairs += pairs
    rem = current % 2
    count_map[x] = rem
    next_size = x * 2
    if next_size in count_map:
        count_map[next_size] += pairs
    else:
        count_map[next_size] = pairs
        heapq.heappush(heap, next_size)

minimal_slimes = sum(count_map.values())
print(minimal_slimes)