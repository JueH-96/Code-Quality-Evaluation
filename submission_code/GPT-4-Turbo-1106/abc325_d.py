import heapq

def max_printable_products(events):
    events.sort()
    heap = []
    count = 0

    for start, end in events:
        while heap and heap[0] < start:
            heapq.heappop(heap)
        if not heap or heap[0] <= start:
            count += 1
            heapq.heappush(heap, end + 1)
        elif heap[0] == start + 1:
            heapq.heappop(heap)
            heapq.heappush(heap, end + 1)

    return count

# Read input
N = int(input().strip())
events = [tuple(map(int, input().split())) for _ in range(N)]

# Calculate and print the result
print(max_printable_products(events))