import heapq

# Read input values
N, K, Q = map(int, input().split())
updates = [tuple(map(int, input().split())) for _ in range(Q)]

# Initialize the sequence A with zeros
A = [0] * N

# Use a max-heap to keep track of the largest K elements
max_heap = []

# Function to update the heap and calculate f(A)
def update_heap(index, value):
    if len(max_heap) < K:
        heapq.heappush(max_heap, value)
    else:
        heapq.heappushpop(max_heap, value)

# Process each update
for X, Y in updates:
    # Update the sequence A
    A[X - 1] = Y

    # Update the heap
    update_heap(X - 1, Y)

    # Calculate f(A)
    f_A = sum(heapq.nlargest(K, max_heap))

    # Print the result
    print(f_A)