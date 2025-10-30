import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
Q = int(data[2])

updates = data[3:]

A = [0] * N

# We will use a max-heap to keep track of the largest K elements
max_heap = []
heap_map = {}

# Initialize the heap with zeros
for i in range(K):
    heapq.heappush(max_heap, 0)

# To keep track of the sum of the top K elements
current_sum = 0

# Process each update
index = 0
for _ in range(Q):
    X = int(updates[index]) - 1
    Y = int(updates[index + 1])
    index += 2
    
    # Update the value at position X
    old_value = A[X]
    A[X] = Y
    
    # If the old value was in the heap, we need to adjust the heap
    if old_value in heap_map and heap_map[old_value] > 0:
        # Decrease the count of the old value in the map
        heap_map[old_value] -= 1
        if heap_map[old_value] == 0:
            del heap_map[old_value]
        
        # Add the new value to the heap
        heapq.heappush(max_heap, Y)
        if Y in heap_map:
            heap_map[Y] += 1
        else:
            heap_map[Y] = 1
        
        # Remove the smallest element if the heap grows larger than K
        while len(max_heap) > K:
            smallest = heapq.heappop(max_heap)
            if heap_map[smallest] > 0:
                heap_map[smallest] -= 1
                if heap_map[smallest] == 0:
                    del heap_map[smallest]
    
    # If the old value was not in the heap, just consider adding the new value
    else:
        # We only add the new value if it's larger than the smallest in the heap
        if Y > max_heap[0]:
            popped = heapq.heappushpop(max_heap, Y)
            if popped in heap_map and heap_map[popped] > 0:
                heap_map[popped] -= 1
                if heap_map[popped] == 0:
                    del heap_map[popped]
            if Y in heap_map:
                heap_map[Y] += 1
            else:
                heap_map[Y] = 1
        else:
            # If not added, we still need to track it
            if Y in heap_map:
                heap_map[Y] += 1
            else:
                heap_map[Y] = 1
    
    # Calculate the sum of the top K elements
    current_sum = sum(max_heap)
    print(current_sum)