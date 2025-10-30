# YOUR CODE HERE
import heapq

def min_cost(N, D, P, F):
    # Sort the fares in descending order
    F.sort(reverse=True)
    
    # Initialize total cost
    total_cost = 0
    
    # Use a heap to manage the minimum fares
    heap = []
    
    # Add the first D fares to the heap
    for i in range(min(D, N)):
        heapq.heappush(heap, F[i])
    
    # Calculate the initial cost with the first batch of passes
    total_cost = P
    
    # Process the remaining days
    for i in range(D, N):
        # If the current fare is less than the smallest fare in the heap, replace it
        if F[i] < heap[0]:
            total_cost += F[i]
        else:
            total_cost += heapq.heappop(heap)
            heapq.heappush(heap, F[i])
    
    # Add the remaining fares in the heap to the total cost
    while heap:
        total_cost += heapq.heappop(heap)
    
    return total_cost

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
P = int(data[2])
F = list(map(int, data[3:]))

# Calculate and print the minimum cost
print(min_cost(N, D, P, F))