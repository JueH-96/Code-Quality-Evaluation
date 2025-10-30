import heapq

def min_slimes():
    N = int(input().strip())
    slimes = []
    for _ in range(N):
        S, C = map(int, input().strip().split())
        slimes.append((S, C))

    slimes.sort()
    heap = []
    for S, C in slimes:
        if heap and heap[0] < S:
            heapq.heappop(heap)
        heapq.heappush(heap, S)
        while C > 1:
            heapq.heappush(heap, S)
            C -= 1

    return len(heap)

print(min_slimes())