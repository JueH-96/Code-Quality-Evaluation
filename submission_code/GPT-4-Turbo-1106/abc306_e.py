import heapq

# Read the input values
N, K, Q = map(int, input().split())
A = [0] * N

# Create a max heap for the first K elements
max_heap = [(-val, idx) for idx, val in enumerate(A[:K])]
heapq.heapify(max_heap)

# Create a dictionary to store the index of the elements in the heap
heap_indices = {idx: idx for idx in range(K)}

# Process the updates and print f(A) after each update
for _ in range(Q):
    X, Y = map(int, input().split())
    X -= 1  # Convert to 0-based index
    A[X] = Y  # Update the sequence

    # If the updated element is in the heap or could enter the heap
    if X < K or A[X] > -max_heap[0][0]:
        if X < K:
            # Update the element in the heap
            heap_indices[X] = len(max_heap)
            heapq.heappush(max_heap, (-A[X], X))
        else:
            # Replace the smallest element in the heap
            _, idx_to_remove = heapq.heappop(max_heap)
            del heap_indices[idx_to_remove]
            heap_indices[X] = len(max_heap)
            heapq.heappush(max_heap, (-A[X], X))

    # Calculate f(A) by summing the K largest elements
    f_A = -sum([val for val, idx in max_heap if idx in heap_indices])
    print(f_A)