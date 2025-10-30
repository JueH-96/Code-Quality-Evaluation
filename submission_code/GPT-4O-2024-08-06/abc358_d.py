# YOUR CODE HERE
import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # Sort the available boxes by candy count
    A.sort()
    # Sort the requirements
    B.sort()
    
    # Min-heap to store the costs of boxes that can satisfy the current requirement
    min_heap = []
    total_cost = 0
    j = 0
    
    for b in B:
        # Add all boxes that can satisfy the current requirement to the heap
        while j < N and A[j] < b:
            j += 1
        while j < N and A[j] >= b:
            heapq.heappush(min_heap, A[j])
            j += 1
        
        # If the heap is empty, we cannot satisfy this requirement
        if not min_heap:
            print(-1)
            return
        
        # Take the cheapest box that satisfies the current requirement
        total_cost += heapq.heappop(min_heap)
    
    print(total_cost)