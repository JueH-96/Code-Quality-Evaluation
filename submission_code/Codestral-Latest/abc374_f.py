import heapq

def min_dissatisfaction(N, K, X, T):
    # Initialize the heap and the current day
    heap = []
    current_day = 0
    total_dissatisfaction = 0

    for i in range(N):
        # If the heap is full, pop the smallest element
        if len(heap) == K:
            heapq.heappop(heap)

        # Push the current order's placement day into the heap
        heapq.heappush(heap, T[i])

        # If the heap is full, calculate the dissatisfaction for the current batch
        if len(heap) == K:
            # The shipment day is the maximum of the current day and the smallest element in the heap
            shipment_day = max(current_day, heap[0])
            for day in heap:
                total_dissatisfaction += shipment_day - day
            # Update the current day to the next possible shipment day
            current_day = shipment_day + X
            heap = []

    # Handle any remaining orders in the heap
    if heap:
        shipment_day = max(current_day, heap[0])
        for day in heap:
            total_dissatisfaction += shipment_day - day

    return total_dissatisfaction

# Read input
N, K, X = map(int, input().split())
T = list(map(int, input().split()))

# Calculate and print the result
result = min_dissatisfaction(N, K, X, T)
print(result)